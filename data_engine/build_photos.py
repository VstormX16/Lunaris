"""Web copies of the spacecraft photos.

"Ay Uzay Araçları Fotoğraflar/kare" (1200x1200) and ".../yatay" (1920x1080) hold the originals, named "<no>_<name>.jpg".
This writes smaller copies named by craft id: frontend/public/spacecraft/square/<id>.jpg (600x600, used for tiles and
list thumbnails) and frontend/public/spacecraft/wide/<id>.jpg (1600x900, used for the detail page). Originals are not touched.
"""
import json
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "Ay Uzay Araçları Fotoğraflar"
OUT = ROOT / "frontend" / "public" / "spacecraft"
SIZES = {"kare": ("square", (600, 600)), "yatay": ("wide", (1600, 900))}


def main():
    ids = {c["no"]: c["id"] for c in json.loads((ROOT / "frontend" / "src" / "data" / "spacecrafts.json").read_text(encoding="utf-8"))}
    for folder, (name, size) in SIZES.items():
        (OUT / name).mkdir(parents=True, exist_ok=True)
        files = sorted((SRC / folder).glob("*.jpg"))
        assert len(files) == len(ids), f"{folder}: {len(files)} photos for {len(ids)} craft"
        for f in files:
            craft_id = ids[int(f.name.split("_")[0])]
            with Image.open(f) as im:
                im = im.convert("RGB")
                assert abs(im.width / im.height - size[0] / size[1]) < 0.01, f"{f.name} is not {name}"
                im.resize(size, Image.LANCZOS).save(OUT / name / f"{craft_id}.jpg", quality=84, optimize=True, progressive=True)
        print(name, len(files), "photos")


if __name__ == "__main__":
    main()
