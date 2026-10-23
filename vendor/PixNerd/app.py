import random
import os
import time
import spaces
import torch
import argparse
from omegaconf import OmegaConf
from src.models.autoencoder.base import fp2uint8
from src.diffusion.base.guidance import simple_guidance_fn
from src.diffusion.flow_matching.adam_sampling import AdamLMSampler
from src.diffusion.flow_matching.scheduling import LinearScheduler
from PIL import Image
import gradio as gr
import tempfile


from huggingface_hub import snapshot_download


def instantiate_class(config):
    kwargs = config.get("init_args", {})
    class_module, class_name = config["class_path"].rsplit(".", 1)
    module = __import__(class_module, fromlist=[class_name])
    args_class = getattr(module, class_name)
    return args_class(**kwargs)

def load_model(weight_dict, denoiser):
    prefix = "ema_denoiser."
    for k, v in denoiser.state_dict().items():
        try:
            v.copy_(weight_dict["state_dict"][prefix + k])
        except:
            print(f"Failed to copy {prefix + k} to denoiser weight")
    return denoiser


class Pipeline:
    def __init__(self, vae, denoiser, conditioner, resolution):
        self.vae = vae
        self.denoiser = denoiser.cuda()
        self.conditioner = conditioner.cuda()
        self.resolution = resolution
        self.tmp_dir = "./traj_gifs_"

        # 若文件夹已存在，先删除旧文件（可选，根据需求决定）
        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)  # 递归删除文件夹及内容
        os.makedirs(self.tmp_dir, exist_ok=False)
        # self.denoiser.compile()


    @spaces.GPU(duration=5)
    @torch.no_grad()
    @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    def __call__(self, y, seed, num_steps, guidance, timeshift, order):
        diffusion_sampler = AdamLMSampler(
            order=order,
            scheduler=LinearScheduler(),
            guidance_fn=simple_guidance_fn,
            num_steps=num_steps,
            guidance=guidance,
            timeshift=timeshift
        )
        generator = torch.Generator(device="cpu").manual_seed(seed)
        xT = torch.randn((1, 3, 512, 512), device="cpu", dtype=torch.float32,
                         generator=generator).cuda()

        start = time.time()
        with torch.no_grad():
            condition, uncondition = conditioner([y,]*1)
        print("conditioner:",time.time() - start)

        start = time.time()
        # Sample images:
        samples, trajs = diffusion_sampler(denoiser, xT, condition, uncondition, return_x_trajs=True)
        print("diffusion:",time.time() - start)

        def decode_images(samples):
            samples = vae.decode(samples)
            samples = fp2uint8(samples)
            samples = samples.permute(0, 2, 3, 1).cpu().numpy()
            images = []
            for i in range(len(samples)):
                image = Image.fromarray(samples[i])
                images.append(image)
            return images

        # def decode_trajs(trajs):
        #     cat_trajs = torch.stack(trajs, dim=0).permute(1, 0, 2, 3, 4)
        #     animations = []
        #     for i in range(cat_trajs.shape[0]):
        #         frames = decode_images(
        #             cat_trajs[i]
        #         )
        #         # 生成唯一文件名（结合seed和样本索引，避免冲突）
        #         gif_filename = f"{random.randint(0, 100000)}.gif"
        #         gif_path = os.path.join(self.tmp_dir, gif_filename)
        #         frames[0].save(
        #             gif_path,
        #             format="GIF",
        #             append_images=frames[1:],
        #             save_all=True,
        #             duration=200,
        #             loop=0
        #         )
        #         animations.append(gif_path)
        #     return animations

        images = decode_images(samples)
        # animations = decode_trajs(trajs)

        return images[0], None #animations[0]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs_t2i/sft_res512.yaml")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--model_id", type=str, default="MCG-NJU/PixNerd-XXL-P16-T2I")
    parser.add_argument("--ckpt_path", type=str, default="models")

    args = parser.parse_args()
    if not os.path.exists(args.ckpt_path):
        snapshot_download(repo_id=args.model_id, local_dir=args.ckpt_path)
        ckpt_path = os.path.join(args.ckpt_path, "model.ckpt")
    else:
        ckpt_path = args.ckpt_path

    config = OmegaConf.load(args.config)
    vae_config = config.model.vae
    denoiser_config = config.model.denoiser
    conditioner_config = config.model.conditioner

    vae = instantiate_class(vae_config)
    denoiser = instantiate_class(denoiser_config)
    conditioner = instantiate_class(conditioner_config)


    ckpt = torch.load(ckpt_path, map_location="cpu")
    denoiser = load_model(ckpt, denoiser)
    denoiser = denoiser.cuda()
    conditioner = conditioner.cuda()
    vae = vae.cuda()
    denoiser.eval()


    pipeline = Pipeline(vae, denoiser, conditioner, args.resolution)

    with gr.Blocks() as demo:
        gr.Markdown(f"config:{args.config}\n\n ckpt_path:{args.ckpt_path}")
        with gr.Row():
            with gr.Column(scale=1):
                num_steps = gr.Slider(minimum=1, maximum=100, step=1, label="num steps", value=25)
                guidance = gr.Slider(minimum=0.1, maximum=10.0, step=0.1, label="CFG", value=4.0)
                label = gr.Textbox(label="positive prompt", value="a photo of a cat")
                seed = gr.Slider(minimum=0, maximum=1000000, step=1, label="seed", value=0)
                timeshift = gr.Slider(minimum=0.1, maximum=5.0, step=0.1, label="timeshift", value=3.0)
                order = gr.Slider(minimum=1, maximum=4, step=1, label="order", value=2)
            with gr.Column(scale=1):
                btn = gr.Button("Generate")
                output_sample = gr.Image(label="Images")
            with gr.Column(scale=1):
                 output_trajs = gr.Image(label="Trajs of Diffusion")

        btn.click(fn=pipeline,
                  inputs=[
                      label,
                      seed,
                      num_steps,
                      guidance,
                      timeshift,
                      order
                  ], outputs=[output_sample, output_trajs])
    demo.launch()