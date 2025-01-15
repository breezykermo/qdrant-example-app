set shell := ["bash", "-uc"]

up:
  source .env
  docker-compose up

down:
  docker-compose down

clean:
  sudo rm -rf data/node1/**
  sudo rm -rf data/node2/**

initialize:
  docker-compose run create_embeddings

example:
  docker-compose run example_search 
