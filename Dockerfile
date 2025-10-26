FROM python:3.9-slim

WORKDIR /app

COPY requirements_prod.txt .
RUN pip install --no-cache-dir -r requirements_prod.txt

COPY ./models/champion_model.pkl ./models/champion_model.pkl

COPY ./src ./src

EXPOSE 8000

CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:8000", "src.api:app"]