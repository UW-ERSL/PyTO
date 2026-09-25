"""Side-by-side images: figures from the papers in Results/Publications next to PyTO's topologies.

Usage (from the repo root, after running the ThermoStructural benchmarks with MMA):
    python -m pyto.examples_benchmarks.make_publication_comparison Results/Results_<date>/ThermoStructural/DENSITYMMA

Writes Results/Publications/PyTO_comparison/{BiClamp_RodriguesFernandes.png, MBBBeam_Ooms.png}. The paper
figures are rasterized from the PDFs with `pdftoppm` (poppler-utils); crop boxes are for 200 dpi renders.
"""
import os
import subprocess
import sys
import tempfile

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

PUB = "Results/Publications"
OUT = os.path.join(PUB, "PyTO_comparison")
RF_PDF = "A material based model for topology optimization of thermoelastic.pdf"
OO_PDF = ("Thermoelastic topology optimization of structural components at elevated temperatures considering "
          "transient heat conduction.pdf")

# (pdf, page, crop box at 200 dpi, label)
PAPER_FIGURES = {
    "BiClamp": [(RF_PDF, 10, (150, 920, 1400, 1880), "Rodrigues & Fernandes (1995), Fig. 4: dT = 0 (b), dT = 1 K (c, d)"),
                (RF_PDF, 11, (150, 1370, 1400, 1800), "Rodrigues & Fernandes (1995), Fig. 6: dT = 4 K (a), dT = 10 K (b)")],
    "MBBBeam": [(OO_PDF, 15, (150, 1430, 1540, 2080), "Ooms et al. (2023), Fig. 6: (a) t_f = 0 (mechanical) ... (i) steady state")],
}
PYTO_PANELS = {
    "BiClamp": ["BiClampDT0", "BiClamp", "BiClampDT4", "BiClampDT10"],
    "MBBBeam": ["MBBBeamNoHeat", "MBBBeam"],
}


def _font(size):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _paper_figure(pdf, page, box, tmp):
    prefix = os.path.join(tmp, f"p{page}")
    subprocess.run(["pdftoppm", "-r", "200", "-f", str(page), "-l", str(page), "-png", os.path.join(PUB, pdf), prefix],
                   check=True)
    png = next(os.path.join(tmp, f) for f in sorted(os.listdir(tmp)) if f.startswith(f"p{page}"))
    return Image.open(png).convert("RGB").crop(box)


def _pyto_panel(result_dir, name, summary):
    img = Image.open(os.path.join(result_dir, f"{name}.png")).convert("RGB")
    img = img.crop(img.convert("L").point(lambda v: 255 - v).getbbox() or (0, 0, *img.size))  # trim white margin
    row = summary[summary["name"] == name]
    caption = name
    if len(row):
        r = row.iloc[0]
        caption = f"{name}: J = {float(r['objective']):.4g}, vf = {float(r['volfrac']):.3f}"
    return img, caption


def _stack(blocks, width):
    """blocks: list of (image, caption). Each image scaled to `width`, caption above it."""
    font = _font(28)
    rows = []
    for img, cap in blocks:
        img = img.resize((width, round(img.height * width / img.width)))
        canvas = Image.new("RGB", (width, img.height + 50), "white")
        ImageDraw.Draw(canvas).text((10, 10), cap, fill="black", font=font)
        canvas.paste(img, (0, 50))
        rows.append(canvas)
    out = Image.new("RGB", (width, sum(r.height for r in rows)), "white")
    y = 0
    for r in rows:
        out.paste(r, (0, y))
        y += r.height
    return out


def _grid(panels, cols, width):
    font = _font(24)
    cell_w = width // cols
    cells = []
    for img, cap in panels:
        img = img.resize((cell_w - 20, round(img.height * (cell_w - 20) / img.width)))
        c = Image.new("RGB", (cell_w, img.height + 45), "white")
        ImageDraw.Draw(c).text((10, 8), cap, fill="black", font=font)
        c.paste(img, (10, 45))
        cells.append(c)
    rows = [cells[i:i + cols] for i in range(0, len(cells), cols)]
    heights = [max(c.height for c in r) for r in rows]
    out = Image.new("RGB", (width, sum(heights)), "white")
    y = 0
    for r, h in zip(rows, heights):
        for k, c in enumerate(r):
            out.paste(c, (k * cell_w, y))
        y += h
    return out


def main(result_dir):
    os.makedirs(OUT, exist_ok=True)
    summary = pd.read_csv(os.path.join(result_dir, "DENSITYMMA_summary.csv"))
    width = 1400
    for problem, figures in PAPER_FIGURES.items():
        with tempfile.TemporaryDirectory() as tmp:
            paper = [(_paper_figure(pdf, page, box, tmp), label) for pdf, page, box, label in figures]
        pyto = _grid([_pyto_panel(result_dir, n, summary) for n in PYTO_PANELS[problem]],
                     cols=2, width=width)
        sheet = _stack(paper + [(pyto, f"PyTO (MMA), {os.path.basename(os.path.dirname(os.path.dirname(result_dir)))}")],
                       width)
        name = {"BiClamp": "BiClamp_RodriguesFernandes.png", "MBBBeam": "MBBBeam_Ooms.png"}[problem]
        sheet.save(os.path.join(OUT, name))
        print("wrote", os.path.join(OUT, name))


if __name__ == "__main__":
    main(sys.argv[1])
