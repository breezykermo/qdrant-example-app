set shell := ["bash", "-uc"]

up:
  source .env
  docker-compose up -d --remove-orphans

down:
  docker-compose down

clean:
  sudo rm -rf data/node1/**
  sudo rm -rf data/node2/**

initialize:
  docker-compose run create_collection 

example:
  echo "TODO:"
