# Full challenge paper source

Expanded from the four-page factsheet into a full ECCV-workshop-style challenge report
(Introduction, Related Work, Method, Experiments, Ablation, Discussion, Conclusion).
Reuses the organizer-provided CVPR-style template (`cvpr.sty`) as no separate ECCV LoViF
paper template was supplied; swap in the official style file if/when organizers provide one.

Compile from this directory with either:

```bash
tectonic AiORestoration_paper.tex
```

or a standard TeX Live installation:

```bash
pdflatex AiORestoration_paper.tex
bibtex AiORestoration_paper
pdflatex AiORestoration_paper.tex
pdflatex AiORestoration_paper.tex
```

Expected output: `AiORestoration_paper.pdf`.

## Before submitting to OpenReview

- Confirm the venue's current style and page-limit requirements, and swap in the official
  template if the organizers provide one.
- Double-check the author list, affiliations, acknowledgements, and any co-author additions.
- Re-verify leaderboard and ablation numbers against the authoritative challenge records before
  submission; private experiment logs and organizer correspondence are intentionally not part of
  this public repository.
