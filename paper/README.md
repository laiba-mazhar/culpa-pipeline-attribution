# Building the paper

`paper.tex` is IEEE conference format (`IEEEtran`, the class IEEE requires for
conference submissions). This directory is self-contained — `paper.tex` and
`main_result.pdf` are all you need.

## Overleaf (easiest — nothing to install)

1. Go to [overleaf.com](https://www.overleaf.com), **New Project → Upload Project**
2. Upload `paper.tex` and `main_result.pdf` (or a zip of this folder)
3. Overleaf ships `IEEEtran` already, so it compiles with no setup
4. Set the compiler to **pdfLaTeX** if it isn't already (Menu → Compiler)

## Locally

Needs a TeX distribution — [MiKTeX](https://miktex.org/download) on Windows.
MiKTeX installs `IEEEtran` on first use when prompted.

```bash
pdflatex paper.tex && pdflatex paper.tex
```

Run it twice: the first pass writes the cross-references, the second resolves
them. There is no `.bib` file — references are inline `\bibitem` entries, so no
BibTeX pass is needed.

## Regenerating the figure

`main_result.pdf` is generated from the experiment CSVs, not drawn by hand, so
it cannot drift out of sync with the results:

```bash
cd .. && python -m experiments.make_figures
```

That writes to `figures/` and to this directory.

## Before submitting anywhere

- Replace the placeholder affiliation in `\author{}`
- Check the target venue's page limit — IEEE conferences are usually 6–8 pages
- Section VIII (Limitations) is deliberately blunt. Keep it. Reviewers find
  these anyway, and stating them first is worth more than hiding them.
