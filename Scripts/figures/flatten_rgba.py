"""Flatten RGBA figures onto a white background, in place.

Elsevier's artwork pipeline does not guarantee how an alpha channel is
composited; a transparent background can be rendered black. Submitted PNGs
should therefore be plain RGB on white. DPI metadata is preserved.

    py -3 flatten_rgba.py [--check]
"""

import glob
import os
import sys

from PIL import Image

Image.MAX_IMAGE_PIXELS = None

HERE = os.path.dirname(os.path.abspath(__file__))
PATTERNS = ["*.png",
            os.path.join("supplementary", "*.png"),
            os.path.join("grayscale", "*.png")]


def main(check_only=False):
    paths = sorted(p for pat in PATTERNS for p in glob.glob(os.path.join(HERE, pat)))
    converted = 0
    for p in paths:
        im = Image.open(p)
        if im.mode not in ("RGBA", "LA", "P"):
            continue
        if im.mode == "P" and "transparency" not in im.info:
            continue
        rel = os.path.relpath(p, HERE)
        if check_only:
            print(f"  would flatten: {rel}  ({im.mode})")
            converted += 1
            continue

        dpi = im.info.get("dpi")
        rgba = im.convert("RGBA")
        bg = Image.new("RGB", rgba.size, (255, 255, 255))
        bg.paste(rgba, mask=rgba.split()[3])
        bg.save(p, dpi=dpi) if dpi else bg.save(p)
        print(f"  flattened: {rel}  {im.mode} -> RGB  dpi={dpi}")
        converted += 1

    verb = "need flattening" if check_only else "flattened"
    print(f"{converted} file(s) {verb}; {len(paths)} scanned")


if __name__ == "__main__":
    main(check_only="--check" in sys.argv)
