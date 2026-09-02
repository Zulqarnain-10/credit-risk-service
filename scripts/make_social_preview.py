"""Render the GitHub social preview card (1280 x 640) from the measured metrics.

The card follows the personal brand spec: navy ground, repo name in Bricolage Grotesque, a Fraunces
italic payoff with a coral underline, one mono metric line with its receipt, and the Z. mark bottom
right. Numbers come from reports/metrics.json only.

Usage:
    python scripts/make_social_preview.py [--out docs/social-preview.png] [--font-dir <dir>]

Fonts are fetched once from the google/fonts repository (OFL licensed) into --font-dir.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
FONT_BASE = "https://github.com/google/fonts/raw/main/ofl/"
FONTS = {
    "bricolage": "bricolagegrotesque/BricolageGrotesque%5Bopsz%2Cwdth%2Cwght%5D.ttf",
    "fraunces_italic": "fraunces/Fraunces-Italic%5BSOFT%2CWONK%2Copsz%2Cwght%5D.ttf",
    "jakarta": "plusjakartasans/PlusJakartaSans%5Bwght%5D.ttf",
    "mono": "ibmplexmono/IBMPlexMono-Regular.ttf",
    "mono_medium": "ibmplexmono/IBMPlexMono-Medium.ttf",
}

NAVY = (10, 22, 40)
FOG = (234, 240, 249)
MIST = (162, 180, 206)
CORAL = (255, 107, 53)
BLUE = (59, 150, 255)
CHIP = (15, 31, 56)

W, H = 1280, 640
SCALE = 2  # render at 2x, downsample for clean edges


def fetch_fonts(font_dir: Path) -> dict[str, Path]:
    font_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for key, rel in FONTS.items():
        target = font_dir / rel.rsplit("/", 1)[-1].replace("%5B", "[").replace("%5D", "]").replace(
            "%2C", ","
        )
        if not target.exists():
            print(f"fetching {rel}")
            urllib.request.urlretrieve(FONT_BASE + rel, target)
        paths[key] = target
    return paths


def font(
    path: Path, size: int, wght: float | None = None, opsz: float | None = None
) -> ImageFont.FreeTypeFont:
    f = ImageFont.truetype(str(path), size * SCALE)
    if wght is not None or opsz is not None:
        try:
            axes = f.get_variation_axes()
            values = []
            for axis in axes:
                raw = axis["name"]
                name = raw.decode() if isinstance(raw, bytes) else str(raw)
                if name.lower() == "weight" and wght is not None:
                    values.append(wght)
                elif name.lower() in ("optical size", "opsz") and opsz is not None:
                    values.append(opsz)
                else:
                    values.append(axis["default"])
            f.set_variation_by_axes(values)
        except OSError:
            pass
    return f


def radial_tint(
    base: Image.Image, center: tuple[int, int], radius: int, color: tuple, alpha: float
) -> None:
    """Paint a soft radial tint onto base (in place)."""
    overlay = Image.new("RGBA", base.size, (*color, 0))
    mask = Image.new("L", base.size, 0)
    draw = ImageDraw.Draw(mask)
    steps = 40
    for i in range(steps, 0, -1):
        r = int(radius * i / steps)
        level = int(255 * alpha * (1 - i / steps) ** 1.6)
        draw.ellipse([center[0] - r, center[1] - r, center[0] + r, center[1] + r], fill=level)
    overlay.putalpha(mask)
    base.alpha_composite(overlay)


def z_mark(draw: ImageDraw.ImageDraw, x: int, y: int, size: int) -> None:
    """The Z. chip mark at (x, y) with the given side length, exact geometry from the brand spec."""
    s = size / 64
    draw.rounded_rectangle([x, y, x + size, y + size], radius=int(14 * s), fill=CHIP)
    pts = [
        (12, 13),
        (44, 13),
        (44, 22.5),
        (25.5, 41.5),
        (44, 41.5),
        (44, 51),
        (12, 51),
        (12, 41.5),
        (30.5, 22.5),
        (12, 22.5),
    ]
    draw.polygon([(x + px * s, y + py * s) for px, py in pts], fill=FOG)
    cx, cy, r = x + 51.5 * s, y + 45.5 * s, 5.5 * s
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=CORAL)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(REPO_ROOT / "docs" / "social-preview.png"))
    ap.add_argument("--font-dir", default=str(Path.home() / ".cache" / "credit-risk-fonts"))
    ap.add_argument("--metrics", default=str(REPO_ROOT / "reports" / "metrics.json"))
    args = ap.parse_args()

    metrics = json.loads(Path(args.metrics).read_text(encoding="utf-8"))
    fonts = fetch_fonts(Path(args.font_dir))

    img = Image.new("RGBA", (W * SCALE, H * SCALE), (*NAVY, 255))
    radial_tint(img, (W * SCALE, 0), int(620 * SCALE), BLUE, 0.16)
    radial_tint(img, (0, H * SCALE), int(560 * SCALE), CORAL, 0.12)
    draw = ImageDraw.Draw(img)

    margin = 72 * SCALE
    kicker_font = font(fonts["mono_medium"], 22)
    head_font = font(fonts["bricolage"], 86, wght=800, opsz=96)
    payoff_font = font(fonts["fraunces_italic"], 86, wght=550, opsz=96)
    support_font = font(fonts["jakarta"], 28, wght=450)
    receipt_font = font(fonts["mono"], 22)

    # Kicker (tracked uppercase mono)
    kicker = "CREDIT-RISK-SERVICE  ·  GITHUB.COM/ZULQARNAIN-10"
    x, y = margin, margin
    for ch in kicker:
        draw.text((x, y), ch, font=kicker_font, fill=MIST)
        x += draw.textlength(ch, font=kicker_font) + 2.2 * SCALE

    # Headline: Bricolage line, then Fraunces italic payoff with a coral underline
    y = 168 * SCALE
    draw.text((margin, y), "Credit-default risk,", font=head_font, fill=FOG)
    y2 = y + 100 * SCALE
    payoff = "shipped like a product."
    draw.text((margin, y2), payoff, font=payoff_font, fill=FOG)
    pw = draw.textlength(payoff, font=payoff_font)
    uy = y2 + 96 * SCALE
    draw.rectangle([margin, uy, margin + pw, uy + 5 * SCALE], fill=CORAL)

    # Support line
    support_lines = [
        "Versioned data, tracked experiments, a tested FastAPI service,",
        "CI that reproduces the numbers, a live endpoint, drift monitoring.",
    ]
    for i, line in enumerate(support_lines):
        draw.text((margin, (396 + 40 * i) * SCALE), line, font=support_font, fill=MIST)

    # Receipt, bottom left: the number and where it comes from
    receipt_lines = [
        f"ROC-AUC {metrics['roc_auc']:.4f}  ·  PR-AUC {metrics['pr_auc']:.4f}  ·  "
        f"held-out test, n = {metrics['n_test']:,}",
        "receipt: reports/metrics.json",
    ]
    base_y = H * SCALE - margin - 60 * SCALE
    for i, line in enumerate(receipt_lines):
        draw.text((margin, base_y + 34 * i * SCALE), line, font=receipt_font, fill=BLUE)

    # Z. mark bottom right
    size = 96 * SCALE
    z_mark(draw, W * SCALE - margin - size, H * SCALE - margin - size, size)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").resize((W, H), Image.LANCZOS).save(out, optimize=True)
    print(f"wrote {out} ({out.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
