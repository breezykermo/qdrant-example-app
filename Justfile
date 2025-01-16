set shell := ["bash", "-uc"]

up:
  source .env
  docker-compose up

down:
  docker-compose down

clean:
  sudo rm -rf data/node1/**
  sudo rm -rf data/node2/**

download_embeddings:
  wget -q -O scripts/create_embeddings/data/points00000000_00100000.pkl "https://www.dropbox.com/scl/fi/yrjdhkcm8xj79lshew3b9/points00000000_00100000.pkl?rlkey=ycxq6tmdhuj4234dxser9uuii&st=jxgmqyws&dl=1"

initialize:
  docker-compose run create_embeddings

example:
  docker-compose run example_search 
