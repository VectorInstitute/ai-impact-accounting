---
title: DIA Footprint Dashboard
emoji: "\U0001F331"
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: apache-2.0
---

# DIA Footprint Dashboard

A public, read-only dashboard for the **Data & Impact Accounting (DIA)** lab. Pick a
base model and see the **cumulative** energy, carbon, and water footprint rolled up
across its whole family (the base plus every derivative trained on top of it).

- **Data:** [DIA-MVP/dia-state-lab-2026](https://huggingface.co/datasets/DIA-MVP/dia-state-lab-2026) (metadata only, no weights)
- **Project / paper:** [VectorInstitute/ai-impact-accounting](https://github.com/VectorInstitute/ai-impact-accounting)

The footprint of each model is recorded in its model card's `dia_report` block and
ingested into the dataset this Space reads.

Interactive lineage graph powered by vis-network (Model Atlas–style click-to-focus).
