# Hugging Face Docker Space for the read-only DIA dashboard.
#
# Built from the *vendored* layout that scripts/deploy_space.py uploads to the
# Space root (not the full source tree):
#   /app.py                 -> FastAPI entrypoint (space/app.py)
#   /requirements.txt       -> light deps, no torch/transformers
#   /ai_impact_accounting/  -> vendored package
#
# To reproduce the Space build locally, stage those files into an empty dir and
# `docker build` there (see scripts/deploy_space.py --help / LAB.md).
FROM python:3.12-slim

WORKDIR /app

# Keep the HF cache on a guaranteed-writable path inside the Space container.
ENV HF_HOME=/tmp/huggingface \
    PIP_NO_CACHE_DIR=1

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY ai_impact_accounting /app/ai_impact_accounting
COPY app.py /app/app.py

ENV DIA_DATASET=DIA-MVP/dia-state-lab-2026
EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
