from PIL import Image
from pathlib import Path

folder = Path("1/images/own")

for file in folder.iterdir():
    if file.suffix.lower() in [".tif", ".tiff"]:
        try:
            img = Image.open(file)
            output = file.with_suffix(".jpg")

            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")
            elif img.mode == "L":
                img = img.convert("RGB")

            img.save(output, "JPEG", quality=95)

            print(f"Converted: {file.name} -> {output.name}")

        except Exception as e:
            print(f"FAILED: {file.name}")
            print(e)