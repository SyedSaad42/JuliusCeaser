#!/usr/bin/env bash
# Runs ON the server, as the `deploy` user. The deploy workflow calls it over SSH:
#
#   printf %s "$TOKEN" | ssh deploy@server "bash /opt/caesars/remote-deploy.sh <image> <tag> <github-user>"
#
# It pulls the new image, restarts the app, waits until it reports healthy,
# and goes back to the previous version if it does not.
set -euo pipefail

IMAGE="$1"      # e.g. ghcr.io/syedsaad42/juliusceaser
TAG="$2"        # the git commit being deployed
REG_USER="$3"

cd /opt/caesars
COMPOSE="docker compose --env-file .env --env-file current.env"

# the registry token arrives on stdin (never on the command line) and is removed again at the end
docker login ghcr.io -u "$REG_USER" --password-stdin > /dev/null
trap 'docker logout ghcr.io > /dev/null 2>&1 || true' EXIT

PREV=""
if [ -f current.env ]; then
  PREV=$(grep '^IMAGE_TAG=' current.env | cut -d= -f2 || true)
fi

write_env() { printf 'IMAGE=%s\nIMAGE_TAG=%s\n' "$IMAGE" "$1" > current.env; }

wait_healthy() {  # about two minutes
  for _ in $(seq 1 40); do
    cid=$($COMPOSE ps -q app 2> /dev/null || true)
    status=none
    if [ -n "$cid" ]; then
      status=$(docker inspect --format '{{.State.Health.Status}}' "$cid" 2> /dev/null || echo none)
    fi
    [ "$status" = "healthy" ] && return 0
    sleep 3
  done
  return 1
}

echo "deploying $TAG (previous: ${PREV:-none})"
write_env "$TAG"
$COMPOSE pull
$COMPOSE up -d --remove-orphans

if wait_healthy; then
  echo "healthy: $TAG is live"
  # keep disk free on the small server: remove every image of ours except the current and previous one
  docker images "$IMAGE" --format '{{.Tag}}' | grep -vxE "$TAG|${PREV:-__none__}" \
    | xargs -r -I{} docker rmi "$IMAGE:{}" > /dev/null 2>&1 || true
  exit 0
fi

echo "ERROR: $TAG did not become healthy. Last app logs:"
$COMPOSE logs --tail 40 app || true
if [ -n "$PREV" ]; then
  echo "rolling back to $PREV"
  write_env "$PREV"
  $COMPOSE up -d
  echo "rolled back to $PREV"
fi
exit 1
