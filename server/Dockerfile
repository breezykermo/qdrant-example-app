FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY ./app .

# Download collection data from Dropbox
RUN apt-get update && apt-get -y install wget
RUN wget -q -O data.json "https://www.dropbox.com/scl/fi/rn5lfieadbyvslh4avtb1/arxiv-snapshot-title-abstract.json?rlkey=az1bmgim0fdmskspfoney1pyb&st=nipl4mxq&dl=1"

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "1.0.0.0", "--port", "8080"]
