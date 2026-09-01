# Limitations and non-claims

- This is an unofficial reproduction and has not been reviewed or endorsed by
  the Shorkie authors.
- The released candidate weight is trained on an independently reconstructed
  165-genome corpus, not the author-distributed corpus. Prefer the official
  requester-pays corpus when it is available to you.
- Overall MLM PPL proximity does not prove equivalent biological utility.
- Repeat-region PPL differs more than the headline aggregate.
- Figure 2 representation analyses remain sensitive to GC/repeat/length
  composition and incomplete interval metadata.
- Figures 3–7 require the supervised 5,215-track model and frozen real datasets.
  Synthetic heads, proxy labels, and MLM variant surprisal are not substitutes.
- Earlier local audit results involving incomplete weight loading, synthetic
  labels, or custom TE proxies are intentionally excluded from this release.
- The package is an alpha release candidate; API and checkpoint schema may
  change before 1.0.
- The v1.1 D-best label combines two separate facts: v1.1 is the routed-L2
  architecture, while D denotes the Adam-state-reset and cosine-learning-rate
  continuation recipe. It is not an official Shorkie version name.
