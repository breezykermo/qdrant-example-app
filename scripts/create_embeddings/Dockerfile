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
RUN if [ "$download_embeddings" = "false" ] ; then wget -q -O data.json "https://www.dropbox.com/scl/fi/rn5lfieadbyvslh4avtb1/arxiv-snapshot-title-abstract.json?rlkey=az1bmgim0fdmskspfoney1pyb&st=nipl4mxq&dl=1"; else echo "Skipping dataset download" ; fi

# Download points so that embeddings don't need to be generated, a process that can take hours.
RUN if [ "$download_embeddings" = "true" ] ; then wget -q -O data/points00000000_00100000.pkl "https://www.dropbox.com/scl/fi/yrjdhkcm8xj79lshew3b9/points00000000_00100000.pkl?rlkey=ycxq6tmdhuj4234dxser9uuii&st=jxgmqyws&dl=1"; else echo "No embeddings downloaded" ; fi

CMD ["python", "main.py"]
