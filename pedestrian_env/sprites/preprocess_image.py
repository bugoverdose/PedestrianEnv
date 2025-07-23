import os
from PIL import Image

def crop_and_save(image_path, output_path):
    img = Image.open(image_path).convert("RGBA")
    bbox = img.getbbox()
    if bbox:
        cropped = img.crop(bbox)
        cropped.save(output_path)
        print(f"Saved: {output_path}")
    else:
        print(f"Skipped (empty): {image_path}")

def process_all_images(root_dir):
    for dirpath, _, filenames in os.walk(root_dir):
        print(dirpath)
        for filename in filenames:
            print(filename)
            if filename.lower().endswith(".png"):
                input_path = os.path.join(dirpath, filename)
                name, ext = os.path.splitext(filename)
                output_path = os.path.join(dirpath, f"{name}_cropped{ext}")
                crop_and_save(input_path, output_path)

def record_aspect_ratios(root_dir, output_txt):
    with open(output_txt, "w") as f:
        for dirpath, _, filenames in os.walk(root_dir):
            for filename in filenames:
                if filename.endswith("_cropped.png"):
                    file_path = os.path.join(dirpath, filename)
                    try:
                        img = Image.open(file_path)
                        width, height = img.size
                        ratio = width / height if height != 0 else 0
                        f.write(f"{file_path},{width}:{height},{ratio:.4f}\n")
                    except Exception as e:
                        print(f"Error with {file_path}: {e}")

def remove_uncropped_pngs(root_dir):
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith(".png") and not filename.endswith("_cropped.png"):
                file_path = os.path.join(dirpath, filename)
                os.remove(file_path)
                print(f"Deleted: {file_path}")

# NOTE: run inside sprites directory
if __name__ == "__main__":
    pass
    # process_all_images("cars")
    # record_aspect_ratios("cars", "car_ratios.txt")
    # remove_uncropped_pngs("cars")
