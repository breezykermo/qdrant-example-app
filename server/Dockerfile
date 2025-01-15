FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY ./app .

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "1.0.0.0", "--port", "8080"]
