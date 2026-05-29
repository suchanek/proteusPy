import os

from PIL import Image
from PIL.Image import Resampling

# Parameters
input_folder = "/Users/egs/repos/proteusPy_priv/Disulfide_Chapter/SpringerBookChapter/Figures"  # change to your PNG folder
output_folder = "/Users/egs/repos/proteusPy_priv/Disulfide_Chapter/SpringerBookChapter/Figures"  # change to your TIFF output folder
scale_factor = 2  # 2x upscale

# Create output folder if needed
os.makedirs(output_folder, exist_ok=True)

# Process all PNG files
for filename in os.listdir(input_folder):
    if filename.lower().endswith(".png"):
        path = os.path.join(input_folder, filename)
        with Image.open(path) as img:
            # Calculate new size
            new_size = (img.width * scale_factor, img.height * scale_factor)
            # Resize using high-quality resampling
            upscaled = img.resize(new_size, Resampling.LANCZOS)
            # Convert to RGB if needed and save as JPG
            base_name = os.path.splitext(filename)[0]
            out_path = os.path.join(output_folder, base_name + ".jpg")
            if upscaled.mode in ("RGBA", "LA") or (
                upscaled.mode == "P" and "transparency" in upscaled.info
            ):
                # Convert to RGB by pasting on white background
                background = Image.new("RGB", upscaled.size, (255, 255, 255))
                if upscaled.mode == "P":
                    upscaled = upscaled.convert("RGBA")
                background.paste(
                    upscaled,
                    mask=upscaled.split()[3] if upscaled.mode == "RGBA" else None,
                )
                background.save(out_path, format="JPEG", quality=95)
            else:
                # Save directly if already in RGB/L mode
                upscaled.save(out_path, format="JPEG", quality=95)
            print(f"Saved: {out_path}")
