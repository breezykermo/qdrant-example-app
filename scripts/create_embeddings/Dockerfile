FROM python:3.11-slim

ARG download_embeddings

WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt
COPY main.py .

# Download filtered version of original arXiv dataset
RUN apt-get update && apt-get -y install wget 
RUN mkdir -p data
# Process to create filtered snapshot of data from https://www.kaggle.com/datasets/Cornell-University/arxiv?resource=download 
# RUN wget https://www.dropbox.com/scl/fi/rbwb5drprqix2yt9vibo6/arxiv-metadata-oai-snapshot.json?rlkey=vjkh438u3v6zp9t8wjiyrh6qm&st=xcsi0uga&dl=1
# RUN time jq -c '[inputs | {title, abstract}]' arxiv-snapshot.json > data.json
RUN wget -q -O data.json "https://www.dropbox.com/scl/fi/o32x38ju2o3cwkou86jo2/arxiv-snapshot-title-abstract.json?rlkey=03vs22ez0rs69sxn5x9ur5mut&st=erry2zt0&dl=1"

CMD ["python", "main.py"]
