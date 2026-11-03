#!/usr/bin/env bash
# Builds swebench-sandbox:repaired (see local/repair/Dockerfile). Run from the repository root in WSL,
# after local/setup_wsl.sh has built swebench-sandbox:latest.
# Wheels in local/repair/wheels.txt are downloaded once from PyPI (cp313, manylinux) into ~/repair-wheels.
set -euo pipefail
cd "$(dirname "$0")/../.."
dl=~/repair-wheels
mkdir -p "$dl"
grep -v '^#' local/repair/wheels.txt | while read -r spec; do
  name=${spec%%==*}; ver=${spec##*==}
  ls "$dl" | grep -qiE "^${name//-/[-_]}-${ver}-" && continue
  docker run --rm -v "$dl":/out swebench-sandbox:latest pip download -q --no-deps --only-binary=:all: \
    --disable-pip-version-check -d /out "$spec"
done
ctx=$(mktemp -d)
trap 'rm -rf "$ctx"' EXIT
mkdir "$ctx/competition" "$ctx/wheels"
cp wheels/*.whl "$ctx/competition/"
cp "$dl"/*.whl "$ctx/wheels/"
cp -r local/repair/shims local/repair/repair_env.py local/repair/Dockerfile "$ctx/"
docker build -q -t swebench-sandbox:repaired "$ctx"
echo "built swebench-sandbox:repaired with $(ls "$dl" | wc -l) extra wheels"
