import os

from PIL import Image
from xml.etree import ElementTree as ET
from pathlib import Path
from copy import deepcopy
import cairosvg

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

def remove_uncropped_pngs(root_dir, ext = ".png"):
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith(ext) and not filename.endswith("_cropped.png"):
                file_path = os.path.join(dirpath, filename)
                os.remove(file_path)
                print(f"Deleted: {file_path}")

def separate_player_svg(input_path, output_path, blank_bg):
    if blank_bg:
        SRC = Path(f"{input_path}/raw_fill_none")
    else:
        SRC = Path(f"{input_path}/raw")

    OUT = Path(output_path)
    OUT.mkdir(exist_ok=True)

    GRID_W = 256
    GRID_H = 384
    COLS = 4
    ROWS = 4
    tileW = GRID_W // COLS
    tileH = GRID_H // ROWS

    ns = {"svg": "http://www.w3.org/2000/svg"}
    ET.register_namespace("", ns["svg"])

    tree = ET.parse(SRC)
    root = tree.getroot()

    g = root.find(".//{http://www.w3.org/2000/svg}g[@id='g3005']")
    if g is None:
        raise SystemExit("g3005 group not found. check id.")

    direction = ["DOWN", "UP", "LEFT", "RIGHT"]
    for r in range(ROWS):
        for c in range(COLS):
            x0 = c * tileW
            y0 = r * tileH

            svg = ET.Element("{%s}svg" % ns["svg"], {
                "version": "1.1",
                "width": str(tileW),
                "height": str(tileH),
                "viewBox": f"{x0} {y0} {tileW} {tileH}",
            })
            svg.append(deepcopy(g))
            if r < 3:
                out = f"player_{direction[r]}_{c}"
            else:
                out = f"player_{direction[r]}_{3 - c}"
            ET.ElementTree(svg).write(OUT / f"{out}.svg", encoding="utf-8", xml_declaration=True)
            svg_to_png(f"{output_path}/{out}.svg", f"{output_path}/{out}.png")
    print("Done:", OUT)

def svg_to_png(input_svg_path, output_png_path):
    cairosvg.svg2png(
        url=input_svg_path,
        write_to=output_png_path,
        background_color=None # keep transparency
    )

# NOTE: run inside sprites directory
if __name__ == "__main__":
    # process_all_images("cars")
    # record_aspect_ratios("cars", "car_ratios.txt")
    # remove_uncropped_pngs("cars")
    pass
    separate_player_svg("retro_character", "player", blank_bg = True)
    process_all_images("player")
    remove_uncropped_pngs("player", ext = ".png")
    remove_uncropped_pngs("player", ext = ".svg")
