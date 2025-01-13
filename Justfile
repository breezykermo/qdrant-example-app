set shell := ["bash", "-uc"]

up:
  source .env
  docker-compose up -d --remove-orphans
  # Necessary on Linux, so that state files are owned by host
  sudo chown -R $(id -u):$(id -g) data/

down:
  docker-compose down

example:
  echo "TODO:"
