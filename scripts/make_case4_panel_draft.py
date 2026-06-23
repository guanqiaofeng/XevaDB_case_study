from pathlib import Path
from PIL import Image, ImageChops, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "results" / "4_paclitaxel" / "figures"
OUT = FIG_DIR / "Figure5_case4_multiomics_draft.png"

FONT_REG = "/System/Library/Fonts/Supplemental/Arial.ttf"
FONT_BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"


def font(size, bold=False):
    return ImageFont.truetype(FONT_BOLD if bold else FONT_REG, size=size)


def crop_white(im, margin=24):
    im = im.convert("RGB")
    bg = Image.new("RGB", im.size, "white")
    diff = Image.eval(ImageChops.difference(im, bg), lambda p: 255 if p > 18 else 0)
    bbox = diff.getbbox()
    if not bbox:
        return im
    left, top, right, bottom = bbox
    left = max(0, left - margin)
    top = max(0, top - margin)
    right = min(im.width, right + margin)
    bottom = min(im.height, bottom + margin)
    return im.crop((left, top, right, bottom))


def fit_to_box(im, box_w, box_h):
    scale = min(box_w / im.width, box_h / im.height)
    new_size = (int(im.width * scale), int(im.height * scale))
    return im.resize(new_size, Image.Resampling.LANCZOS)


def paste_panel(canvas, path, xy, size, label, title=None, crop=True):
    x, y = xy
    w, h = size
    draw = ImageDraw.Draw(canvas)
    draw.text((x, y), f"{label}.", fill=(0, 0, 0), font=font(54, True))
    if title:
        draw.text((x + 80, y + 4), title, fill=(20, 20, 20), font=font(42, True))

    im = Image.open(path).convert("RGB")
    if crop:
        # Keep titles/labels but remove excessive external whitespace.
        bg = Image.new("RGB", im.size, "white")
        diff = ImageChops.difference(im, bg)
        bbox = diff.getbbox()
        if bbox:
            pad = 25
            im = im.crop((
                max(0, bbox[0] - pad),
                max(0, bbox[1] - pad),
                min(im.width, bbox[2] + pad),
                min(im.height, bbox[3] + pad),
            ))
    fitted = fit_to_box(im, w - 40, h - 115)
    px = x + (w - fitted.width) // 2
    py = y + 95 + (h - 115 - fitted.height) // 2
    canvas.paste(fitted, (px, py))


def trim_nonwhite(im, pad=25):
    im = im.convert("RGB")
    bg = Image.new("RGB", im.size, "white")
    diff = ImageChops.difference(im, bg)
    bbox = diff.getbbox()
    if not bbox:
        return im
    return im.crop((
        max(0, bbox[0] - pad),
        max(0, bbox[1] - pad),
        min(im.width, bbox[2] + pad),
        min(im.height, bbox[3] + pad),
    ))


def paste_roc_only(canvas, path, xy, size, label, title):
    x, y = xy
    w, h = size
    draw = ImageDraw.Draw(canvas)
    draw.text((x, y), f"{label}.", fill=(0, 0, 0), font=font(54, True))
    draw.text((x + 80, y + 4), title, fill=(20, 20, 20), font=font(42, True))

    im = trim_nonwhite(Image.open(path).convert("RGB"))
    mid = im.width // 2
    roc = trim_nonwhite(im.crop((0, 0, mid + 45, im.height)), pad=8)
    fitted = fit_to_box(roc, w - 40, h - 115)
    px = x + (w - fitted.width) // 2
    py = y + 95 + (h - 115 - fitted.height) // 2
    canvas.paste(fitted, (px, py))


def draw_arrow(draw, start, end, fill, width=8):
    draw.line([start, end], fill=fill, width=width)
    ex, ey = end
    sx, sy = start
    dx, dy = ex - sx, ey - sy
    if abs(dx) >= abs(dy):
        pts = [(ex, ey), (ex - 28, ey - 18), (ex - 28, ey + 18)] if dx > 0 else [(ex, ey), (ex + 28, ey - 18), (ex + 28, ey + 18)]
    else:
        pts = [(ex, ey), (ex - 18, ey - 28), (ex + 18, ey - 28)] if dy > 0 else [(ex, ey), (ex - 18, ey + 28), (ex + 18, ey + 28)]
    draw.polygon(pts, fill=fill)


def centered_text(draw, box, text, text_font, fill=(25, 25, 25), spacing=8):
    x0, y0, x1, y1 = box
    words = text.split()
    lines = []
    line = ""
    max_w = x1 - x0 - 36
    for word in words:
        test = f"{line} {word}".strip()
        if draw.textbbox((0, 0), test, font=text_font)[2] <= max_w:
            line = test
        else:
            if line:
                lines.append(line)
            line = word
    if line:
        lines.append(line)
    line_h = text_font.size + spacing
    total_h = len(lines) * line_h - spacing
    y = y0 + ((y1 - y0) - total_h) / 2
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=text_font)
        draw.text((x0 + ((x1 - x0) - (bbox[2] - bbox[0])) / 2, y), line, font=text_font, fill=fill)
        y += line_h


def workflow_panel(canvas, xy, size):
    x, y = xy
    w, h = size
    draw = ImageDraw.Draw(canvas)
    draw.text((x, y), "a.", fill=(0, 0, 0), font=font(54, True))
    draw.text((x + 80, y + 4), "Multi-omics paclitaxel response modeling workflow", fill=(20, 20, 20), font=font(42, True))

    top = y + 120
    box_h = 170
    colors = {
        "blue": (76, 114, 176),
        "teal": (85, 168, 160),
        "orange": (221, 132, 82),
        "red": (196, 78, 82),
        "gray": (100, 100, 100),
        "light": (247, 249, 252),
    }

    boxes = [
        (x + 90, top, x + 640, top + box_h, "Paclitaxel-treated PDX models"),
        (x + 820, top, x + 1370, top + box_h, "TGI / angle / mRECIST response endpoint"),
        (x + 1550, top - 55, x + 2050, top + 55, "RNA-seq"),
        (x + 1550, top + 95, x + 2050, top + 205, "CNV"),
        (x + 1550, top + 245, x + 2050, top + 355, "Mutation"),
        (x + 2270, top - 35, x + 2810, top + 95, "Individual omics models"),
        (x + 2270, top + 150, x + 2810, top + 280, "Early fusion: concat features"),
        (x + 2270, top + 335, x + 2810, top + 465, "Late fusion: stacking predictions"),
        (x + 3040, top + 105, x + 3650, top + 275, "Cross-validated sensitive vs resistant prediction"),
    ]

    for i, (x0, y0, x1, y1, txt) in enumerate(boxes):
        fill = colors["light"]
        outline = colors["gray"]
        if txt == "RNA-seq":
            outline = colors["blue"]
        elif txt == "CNV":
            outline = colors["teal"]
        elif txt == "Mutation":
            outline = colors["orange"]
        elif txt.startswith("Early"):
            outline = colors["red"]
            fill = (255, 246, 246)
        draw.rounded_rectangle((x0, y0, x1, y1), radius=22, fill=fill, outline=outline, width=5)
        centered_text(draw, (x0, y0, x1, y1), txt, font(34, True if i in (0, 1, 8) else False))

    draw_arrow(draw, (x + 640, top + 85), (x + 820, top + 85), colors["gray"])
    draw_arrow(draw, (x + 1370, top + 85), (x + 1550, top + 85), colors["gray"])
    for sy in [top, top + 150, top + 300]:
        draw_arrow(draw, (x + 2050, sy + 55), (x + 2270, sy + 55), colors["gray"])
    for sy in [top + 30, top + 215, top + 400]:
        draw_arrow(draw, (x + 2810, sy + 35), (x + 3040, top + 190), colors["gray"], width=6)

    callout = "Key comparison: RNA-only, CNV-only, mutation-only, early fusion, and late-fusion stacking"
    draw.text((x + 90, y + h - 86), callout, fill=(55, 55, 55), font=font(34))


def main():
    canvas = Image.new("RGB", (3900, 5350), "white")
    workflow_panel(canvas, (90, 60), (3720, 710))

    paste_panel(canvas, FIG_DIR / "response_endpoint_qc.png", (90, 820), (1830, 1370), "b", "Response endpoint validation")
    paste_panel(canvas, FIG_DIR / "angle_vs_tgi_scatter.png", (2070, 820), (1740, 1370), "c", "Angle/TGI concordance")
    paste_panel(canvas, FIG_DIR / "model_benchmark_multi_metric.png", (90, 2290), (3720, 1160), "d", "Model performance across individual omics and fusion strategies")
    paste_roc_only(canvas, FIG_DIR / "roc_pr_curves_representative_models.png", (90, 3560), (1900, 1630), "e", "Representative ROC curves")
    paste_panel(canvas, FIG_DIR / "early_fusion_rf_prediction_scores.png", (2130, 3560), (1680, 1630), "f", "Early-fusion RF prediction scores")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUT, dpi=(300, 300), quality=95)
    print(OUT)


if __name__ == "__main__":
    main()
