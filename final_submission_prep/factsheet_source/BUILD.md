# Factsheet source

The source preserves the organizer-provided CVPR-style template files.

Compile from this directory with either:

```bash
tectonic AiORestoration_factsheet.tex
```

or a standard TeX Live installation:

```bash
pdflatex AiORestoration_factsheet.tex
bibtex AiORestoration_factsheet
pdflatex AiORestoration_factsheet.tex
pdflatex AiORestoration_factsheet.tex
```

Expected output: `AiORestoration_factsheet.pdf`, four US-letter pages.

Before final compilation, replace the Codabench username, OpenReview ID, and public result-archive download-link placeholders in the TeX source.
