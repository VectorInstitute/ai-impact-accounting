FROM python:3.12-slim

WORKDIR /app

COPY space/requirements.txt /app/requirements.txt
COPY pyproject.toml /app/pyproject.toml
COPY src /app/src

RUN pip install --no-cache-dir -r /app/requirements.txt \
    && pip install --no-cache-dir -e "/app[dashboard]"

COPY space/app.py /app/app.py

ENV DIA_DATASET=DIA-MVP/dia-state-lab-2026
EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
