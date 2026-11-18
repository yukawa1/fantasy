"""
Build the README banner: the FANTASY mark, the wordmark FANTASY² (the ² in the warm colours of the
logo, as in the app) and the full name, in a light and a dark version (docs/fantasy2_banner*.png).
Run from the repository root on macOS (it uses Helvetica Neue):

    python tools/build_readme_banner.py
"""
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
MARK = os.path.join(HERE, "..", "fantasy_agn", "app", "static", "img", "mark@2x.png")
OUT = os.path.join(HERE, "..", "docs", "fantasy2_banner{}.png")
FONT = "/System/Library/Fonts/HelveticaNeue.ttc"
S = 2                                                   # drawn at 2x for sharp text on retina screens
TAGLINE = "Fully Automated pythoN Tool for AGN Spectra analYsis"


def font(size, index):
    return ImageFont.truetype(FONT, size * S, index=index)


def banner(ink, teal, name):
    W, H = 860 * S, 170 * S
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    mark = Image.open(MARK).convert("RGBA")
    mh = 132 * S
    mark = mark.resize((round(mark.width * mh / mark.height), mh), Image.LANCZOS)
    img.alpha_composite(mark, (12 * S, (H - mh) // 2))
    d = ImageDraw.Draw(img)
    x0, y0 = 12 * S + mark.width + 28 * S, 34 * S
    big = font(64, 10)                                  # Helvetica Neue Medium
    x = x0
    for ch in "FANTASY":                                # letter-spaced, as the app's wordmark
        d.text((x, y0), ch, font=big, fill=ink)
        x += d.textlength(ch, font=big) + 22 * S
    # the ², filled with the red-orange gradient of the logo's ribbon
    sq = font(40, 1)                                    # bold
    box = d.textbbox((0, 0), "2", font=sq)
    w, h = box[2] - box[0] + 2 * S, box[3] - box[1] + 2 * S
    m = Image.new("L", (w, h), 0)
    ImageDraw.Draw(m).text((-box[0] + S, -box[1] + S), "2", font=sq, fill=255)
    t = np.linspace(0, 1, h)[:, None] * 0.75 + np.linspace(0, 1, w)[None, :] * 0.25
    stops = np.array([[214, 40, 90], [238, 106, 76], [244, 176, 74]], float)
    rgb = np.where(t[..., None] < 0.5, stops[0] + (stops[1] - stops[0]) * (t[..., None] / 0.5),
                   stops[1] + (stops[2] - stops[1]) * ((t[..., None] - 0.5) / 0.5))
    grad = Image.fromarray(np.dstack([rgb, np.asarray(m, float)]).astype(np.uint8), "RGBA")
    img.alpha_composite(grad, (round(x - 14 * S), y0 - 8 * S))
    d.text((x0 + 2 * S, y0 + 86 * S), TAGLINE, font=font(24, 0), fill=teal)
    bbox = img.getbbox()
    img = img.crop((0, 0, min(W, bbox[2] + 12 * S), H))
    img.save(OUT.format(name), optimize=True)
    print(OUT.format(name), img.size)


banner((58, 61, 66, 255), (26, 154, 168, 255), "")         # light pages
banner((230, 232, 236, 255), (82, 196, 208, 255), "_dark")  # dark pages
