"""Fundus photograph preprocessing shared by all datasets."""
import numpy as np
from PIL import Image


def crop_and_pad(img, size=224):
    """Crop to the fundus, pad to a square with black and resize.

    The fundus is found as the bounding box of pixels brighter than a small
    threshold (the background of fundus photographs is close to black).
    """
    img = img.convert("RGB")
    small = img.copy()
    small.thumbnail((512, 512))
    mask = np.asarray(small).astype(np.int16).sum(axis=2) > 30
    if mask.sum() > 0.05 * mask.size:
        ys, xs = np.where(mask)
        sx, sy = img.width / small.width, img.height / small.height
        img = img.crop((int(xs.min() * sx), int(ys.min() * sy),
                        int((xs.max() + 1) * sx), int((ys.max() + 1) * sy)))
    w, h = img.size
    side = max(w, h)
    canvas = Image.new("RGB", (side, side), (0, 0, 0))
    canvas.paste(img, ((side - w) // 2, (side - h) // 2))
    return canvas.resize((size, size), Image.BICUBIC)
