"""Package the paper as a self-contained, Overleaf-ready zip.

IEEEtran.cls is bundled (LPPL, freely redistributable) so the archive compiles on
any TeX installation, not only on Overleaf where the class already ships. Keeping
the archive flat matters: Overleaf picks the main document from the top-level
.tex, and a nested folder makes that guess wrong.

Run:  python -m experiments.make_paper_zip
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
ZIP_NAME = "culpa-paper-ieee.zip"

FIGURES = [
    "main_result.pdf",
    "pipeline_taxi.pdf",
    "attribution.pdf",
    "order_dependence.pdf",
    "severity_ladder.pdf",
    "cost_scaling.pdf",
]

# Flat archive: name in zip -> source path.
CONTENTS = {
    "paper.tex": PAPER / "paper.tex",
    "README.md": PAPER / "README.md",
    "IEEEtran.cls": PAPER / "IEEEtran.cls",
    **{f: PAPER / f for f in FIGURES},
}


def main() -> None:
    missing = [n for n, p in CONTENTS.items() if not p.exists()]
    if missing:
        raise SystemExit(f"missing sources: {missing}\nrun experiments.make_figures first")

    out = PAPER / ZIP_NAME
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for name, path in CONTENTS.items():
            z.write(path, arcname=name)

    with zipfile.ZipFile(out) as z:
        bad = z.testzip()
        if bad is not None:
            raise SystemExit(f"archive is corrupt at {bad}")
        listing = z.infolist()

    print(f"wrote {out}  ({out.stat().st_size / 1024:.1f} KB)\n")
    for i in listing:
        print(f"  {i.filename:20s} {i.file_size / 1024:8.1f} KB")

    print("\nOverleaf: New Project -> Upload Project -> this zip.")
    print("IEEEtran.cls is bundled, so it also compiles with a bare pdflatex.")


if __name__ == "__main__":
    main()
