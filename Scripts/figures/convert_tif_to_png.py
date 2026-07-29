from PIL import Image
import os
import sys

import os

# --- Repository-relative path resolution -------------------------------------
# Figure output goes to PAPER_FIGURES_DIR when set, otherwise to the repository
# Figures/ directory.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
source = os.environ.get('PAPER_FIGURES_DIR', os.path.join(_REPO_ROOT, 'Figures'))
import os

# --- Repository-relative path resolution -------------------------------------
# Figure output goes to PAPER_FIGURES_DIR when set, otherwise to the repository
# Figures/ directory.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
dest = os.environ.get('PAPER_FIGURES_DIR', os.path.join(_REPO_ROOT, 'Figures'))

print(f"Attempting to convert {source} to {dest}...")

if not os.path.exists(source):
    print(f"Error: Source file not found at {source}")
    sys.exit(1)

try:
    with Image.open(source) as img:
        if img.mode == 'CMYK':
            img = img.convert('RGB')
        img.save(dest, "PNG")
    print(f"Successfully converted {source} to {dest}")
except ImportError:
    print("Error: PIL (Pillow) is not installed. Please install it with 'pip install Pillow'")
    sys.exit(1)
except Exception as e:
    print(f"Error during conversion: {e}")
    sys.exit(1)
