# CR-V development and release

`crv-sng-tuning-source` is the full source branch. It carries build files,
submodules, and source-level CR-V changes. `ccrome/openpilot:crv-sng-tuning`
is deliberately separate: it is the stripped, installable Comma release.

For local development on Ubuntu:

```bash
./tools/op.sh setup
source .venv/bin/activate
scons -u
```

The setup script may ask for sudo only to install the optional Panda/Comma USB
udev rule. User-space dependencies can be installed with `uv sync --frozen
--all-extras` when that rule is not needed.

Hardware-independent tests and replay development belong on the workstation.
The final Comma release must be built natively on a powered Comma running
AGNOS. That environment supplies the Qualcomm camera headers, GPU device,
drivers, and runtime libraries needed for a complete build.

Connect with SSH agent forwarding, check out the full source branch in a
separate directory, build there, and verify the compiled parameter registry
and hardware processes before publishing a stripped installer branch. Do not
use local x86 or generic cloud ARM build output in the installer tree.

The release helper accepts a separate build worktree and destination remote:

```bash
BUILD_DIR=/data/crv-sunnypilot-release \
PUBLISH_REMOTE=installer \
PANDA_DEBUG_BUILD=1 \
RUN_ONROAD_TEST=0 \
RELEASE_BRANCH=crv-sng-tuning \
tools/release/build_release.sh
```

`test_onroad.py` deletes its configured log root. Run it only on a dedicated
test device or set `LOG_ROOT` to explicitly disposable storage. On a personal
device, publish with `RUN_ONROAD_TEST=0`, install the resulting prebuilt, and
perform the runtime smoke test without deleting retained drive logs.
