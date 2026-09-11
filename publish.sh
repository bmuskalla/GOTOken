#!/bin/sh
# Build the image for linux/amd64 and linux/arm64 and push it to Docker Hub.
#
#   DOCKERHUB_USERNAME=you DOCKERHUB_TOKEN=dckr_pat_... ./publish.sh [version]
#
# Pushes <username>/gotoken:latest, plus <username>/gotoken:<version> when a
# version is given. DOCKER_IMAGE overrides the image name. The token is a
# Docker Hub personal access token with read/write scope.
set -e
cd "$(dirname "$0")"
: "${DOCKERHUB_USERNAME:?set DOCKERHUB_USERNAME}"
: "${DOCKERHUB_TOKEN:?set DOCKERHUB_TOKEN}"
IMAGE="${DOCKER_IMAGE:-$DOCKERHUB_USERNAME/gotoken}"
TAGS="-t $IMAGE:latest"
[ -n "$1" ] && TAGS="$TAGS -t $IMAGE:$1"

printf '%s' "$DOCKERHUB_TOKEN" | docker login --username "$DOCKERHUB_USERNAME" --password-stdin
docker buildx create --name gotoken-builder --use >/dev/null 2>&1 || docker buildx use gotoken-builder
# shellcheck disable=SC2086
docker buildx build --platform linux/amd64,linux/arm64 $TAGS --push .
echo "pushed $IMAGE (latest${1:+, $1})"
