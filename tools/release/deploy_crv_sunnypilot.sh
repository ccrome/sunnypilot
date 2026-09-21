#!/usr/bin/env bash
set -Eeuo pipefail

# Build and publish the CR-V SunnyPilot release on the Comma.
# This script never modifies the active /data/openpilot installation.

SOURCE_DIR="${SOURCE_DIR:-}"
SOURCE_BRANCH="${SOURCE_BRANCH:-crv-brake-gas-crossover}"
INSTALLER_BRANCH="${INSTALLER_BRANCH:-crv-sng-tuning}"
COMMA_HOST="${COMMA_HOST:-comma@192.168.86.31}"
DEVICE_SOURCE_DIR="${DEVICE_SOURCE_DIR:-/data/crv-sunnypilot-source}"
BUILD_DIR="${BUILD_DIR:-/data/crv-sunnypilot-release}"
SCRIPT_NAME="tools/release/deploy_crv_sunnypilot.sh"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

require_clean_repo() {
  local repo="$1"
  test "$(git -C "$repo" branch --show-current)" = "$SOURCE_BRANCH" \
    || die "$repo is not on $SOURCE_BRANCH"
  test -z "$(git -C "$repo" status --porcelain)" \
    || die "$repo has uncommitted changes"
}

device_build() {
  local source_dir
  source_dir="$(cd "$(dirname "$0")/../.." && pwd)"

  profiles="$(nmcli -t -f NAME,TYPE connection show)"
  grep -Fqx 'openpilot connection badkitty:802-11-wireless' <<<"$profiles" \
    || die 'missing Wi-Fi profile: openpilot connection badkitty'
  grep -Fqx 'sunnypilot connection smudge:802-11-wireless' <<<"$profiles" \
    || die 'missing Wi-Fi profile: sunnypilot connection smudge'
  nmcli connection modify "openpilot connection badkitty" \
    connection.autoconnect yes connection.autoconnect-priority 100
  nmcli connection modify "sunnypilot connection smudge" \
    connection.autoconnect yes connection.autoconnect-priority 90
  nmcli -t -f NAME,TYPE,AUTOCONNECT,AUTOCONNECT-PRIORITY,DEVICE connection show

  test -z "$(git -C "$source_dir" status --porcelain)" \
    || die "$source_dir has uncommitted changes"
  git -C "$source_dir" fetch origin "$SOURCE_BRANCH"
  git -C "$source_dir" checkout "$SOURCE_BRANCH"
  git -C "$source_dir" pull --ff-only origin "$SOURCE_BRANCH"
  git -C "$source_dir" submodule update --init --recursive
  git -C "$source_dir" lfs pull

  test -f "$source_dir/msgq_repo/site_scons/site_tools/cython.py" \
    || die 'source submodules are incomplete: cython.py is missing'
  test "$(git -C "$source_dir" remote get-url installer)" \
    = 'git@github.com:ccrome/openpilot.git' \
    || die 'installer remote is missing or incorrect'
  grep -q 'BUILD_DIR="\${BUILD_DIR:-/data/openpilot}"' \
    "$source_dir/tools/release/build_release.sh" \
    || die 'release helper has no safe BUILD_DIR default'
  grep -q SCONS_JOBS "$source_dir/tools/release/build_release.sh" \
    || die 'release helper does not support bounded SCons jobs'
  test -x /usr/local/venv/bin/scons || die 'Comma SCons is unavailable'

  echo "SOURCE=$(git -C "$source_dir" rev-parse HEAD)"
  echo "OPENDBC=$(git -C "$source_dir/opendbc_repo" rev-parse HEAD)"
  cd "$source_dir"
  PATH=/usr/local/venv/bin:$PATH \
    BUILD_DIR="$BUILD_DIR" \
    PUBLISH_REMOTE=installer \
    PANDA_DEBUG_BUILD=1 \
    RUN_ONROAD_TEST=0 \
    RELEASE_BRANCH="$INSTALLER_BRANCH" \
    SCONS_JOBS=1 \
    tools/release/build_release.sh

  cd "$BUILD_DIR"
  test -e prebuilt
  test ! -e .gitmodules
  test -z "$(git lfs ls-files)"
  test -f openpilot/common/libparams_c.so
  test -x openpilot/system/camerad/camerad
  file openpilot/common/libparams_c.so openpilot/system/camerad/camerad
  PYTHONPATH="$BUILD_DIR:$BUILD_DIR/openpilot:$BUILD_DIR/opendbc_repo:$BUILD_DIR/msgq_repo" \
    /usr/local/venv/bin/python3 -c \
    'from openpilot.common.params import Params; Params().check_key("HondaCrvSpeedScale")'
  if grep -Rqs HondaCrvLongitudinalTuningProfile openpilot opendbc_repo 2>/dev/null; then
    die 'legacy profile parameter is present'
  fi

  locationd="$BUILD_DIR/openpilot/sunnypilot/selfdrive/locationd/locationd"
  live_dir="$BUILD_DIR/openpilot/sunnypilot/selfdrive/locationd/models/generated"
  test -x "$locationd"
  test -f "$live_dir/liblive.so"
  if readelf -d "$locationd" | grep -F "$BUILD_DIR/"; then
    die 'locationd contains a disposable build-path RUNPATH'
  fi
  if LD_LIBRARY_PATH="$live_dir" ldd "$locationd" | grep -F 'not found'; then
    die 'locationd has an unresolved library'
  fi
  echo "RELEASE=$(git rev-parse HEAD)"
  echo ARTIFACT_CHECKS=PASS
}

if [[ "${1:-}" == --device-build ]]; then
  shift
  device_build "$@"
  exit 0
fi

SOURCE_DIR="${SOURCE_DIR:-$(git rev-parse --show-toplevel)}"
OPENDBC_DIR="$SOURCE_DIR/opendbc_repo"
require_clean_repo "$SOURCE_DIR"
require_clean_repo "$OPENDBC_DIR"
git -C "$SOURCE_DIR" diff --check
git -C "$OPENDBC_DIR" diff --check

source_commit="$(git -C "$SOURCE_DIR" rev-parse HEAD)"
opendbc_commit="$(git -C "$OPENDBC_DIR" rev-parse HEAD)"
echo "sunnypilot: $source_commit"
echo "opendbc:    $opendbc_commit"

git -C "$OPENDBC_DIR" push origin "$SOURCE_BRANCH"
git -C "$SOURCE_DIR" push origin "$SOURCE_BRANCH"
git -C "$SOURCE_DIR" fetch origin "$SOURCE_BRANCH"
test "$source_commit" = "$(git -C "$SOURCE_DIR" rev-parse "origin/$SOURCE_BRANCH")"
git -C "$OPENDBC_DIR" fetch origin "$SOURCE_BRANCH"
test "$opendbc_commit" = "$(git -C "$OPENDBC_DIR" rev-parse "origin/$SOURCE_BRANCH")"

ssh -A -o ConnectTimeout=10 "$COMMA_HOST" bash -s -- \
  "$DEVICE_SOURCE_DIR" "$SOURCE_BRANCH" <<'REMOTE_SETUP'
set -Eeuo pipefail
device_source="$1"
source_branch="$2"
if test ! -d "$device_source/.git"; then
  git clone --recurse-submodules --branch "$source_branch" --single-branch \
    https://github.com/ccrome/sunnypilot.git "$device_source"
fi
mkdir -p "$device_source/tools/release"
REMOTE_SETUP

# Keep the device-side copy synchronized whenever this source-tree script changes.
rsync -av --checksum --chmod=F755 \
  "$SOURCE_DIR/$SCRIPT_NAME" \
  "$COMMA_HOST:$DEVICE_SOURCE_DIR/$SCRIPT_NAME"

ssh -A -o ConnectTimeout=10 "$COMMA_HOST" \
  "bash '$DEVICE_SOURCE_DIR/$SCRIPT_NAME' --device-build"

expected="$(ssh -A -o ConnectTimeout=10 "$COMMA_HOST" \
  "cd '$BUILD_DIR' && git rev-parse HEAD")"
actual="$(gh api "repos/ccrome/openpilot/branches/$INSTALLER_BRANCH" --jq .commit.sha)"
echo "published expected: $expected"
echo "published actual:   $actual"
test "$expected" = "$actual"
echo "DEPLOY_OK=$actual"
