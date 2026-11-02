#!/usr/bin/env bash
# From-zero environment setup for UMI_collection. Creates the two uv-managed
# venvs (uv auto-downloads the right CPython), installs pinned deps, and seeds
# the writable per-rig Bota payload from its template.
#
#   ./bootstrap.sh            # create/refresh both venvs + seed payload
#   ./bootstrap.sh --viz-only # only the py3.12 viz venv (no tracker/camera hw)
#
# After this:
#   ./run_collection_viz.sh --mock          # UI-only smoke (no hardware)
#   ./run_collection_viz.sh                 # full stack
#
# One-time host setup NOT done here (documented in docs/HARDWARE_SETUP.md):
#   sudo usermod -aG input "$USER"          # evdev keyboard bridge (re-login)
#   # can0 bring-up is auto-done by run_collection_viz.sh (sudo).
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VIZ_ONLY=""
for a in "$@"; do [ "$a" = "--viz-only" ] && VIZ_ONLY="1"; done

# 0) uv (installs to ~/.local/bin if missing).
if ! command -v uv >/dev/null 2>&1; then
  echo "[bootstrap] installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
echo "[bootstrap] uv $(uv --version)"

# 1) py3.12 venv: viz + IK + bota + gripper + keyboard + UMI conversion.
#    DECISION-I: ONE maintained env does BOTH live collection AND offline
#    conversion (raw recording -> UMI Zarr). The conversion deps (zarr/numcodecs/
#    imagecodecs/av/cv2) install into the SAME .venv — no separate .venv-proc.
echo "[bootstrap] creating .venv (python 3.12)"
uv venv --python 3.12 "$ROOT/.venv"
uv pip install --python "$ROOT/.venv/bin/python" -r "$ROOT/env/requirements-venv.txt"
# dev/test deps (pytest) so the documented test commands run on a fresh checkout.
uv pip install --python "$ROOT/.venv/bin/python" -r "$ROOT/env/requirements-dev.txt"
# Offline UMI-conversion deps (recording -> Zarr ReplayBuffer). In the SAME venv.
uv pip install --python "$ROOT/.venv/bin/python" -r "$ROOT/env/requirements-processing.txt"

# 2) py3.9 venv: the RealSense camera bridge (cp39-locked pyrealsense2 wheel) + the
#    tracker bridge's numpy/websockets. pysurvive is NOT pip-installed:
#    the PyPI wheel (1.1.21, uploaded 2023-03-11, bundling libsurvive "1.2.7" and its
#    own libusb) is three years behind on Gen2-lighthouse OOTX/tracking fixes and its
#    libusb segfaults on some hosts. libsurvive is built FROM SOURCE at the pinned
#    commit (step 2a) and its ctypes bindings — pure Python, interpreter-agnostic — are
#    put on PYTHONPATH by run_collection_viz.sh for the tracker process only. The bridge
#    refuses the wheel by provenance even if one is present.
if [ -z "$VIZ_ONLY" ]; then
  echo "[bootstrap] creating .venv-vive (python 3.9)"
  uv venv --python 3.9 "$ROOT/.venv-vive"
  VIVE_PY="$ROOT/.venv-vive/bin/python"
  uv pip install --python "$VIVE_PY" \
    pyrealsense2==2.56.5.9235 opencv-python==4.13.0.92 numpy==2.0.2 websockets==15.0.1
  # An OLD venv may still carry the wheel; PYTHONPATH would shadow it, but remove it so a
  # bare `import pysurvive` (no env) fails loudly instead of importing 2023 code.
  uv pip uninstall --python "$VIVE_PY" pysurvive >/dev/null 2>&1 || true

  # 2a) libsurvive from source at the pinned commit
  #     5588cadf (2026-05-14), system libusb, `make` in the checkout -> bin/libsurvive.so
  #     + bin/plugins/ + bindings/python/pysurvive. Idempotent: skipped when the build at
  #     that commit already exists. LIBSURVIVE_DIR overrides the location.
  LIBSURVIVE_DIR="${LIBSURVIVE_DIR:-$HOME/libsurvive}"
  LIBSURVIVE_COMMIT="5588cadf5ab764b51e50f9b08996009d1b9f836e"
  if [ -f "$LIBSURVIVE_DIR/bin/libsurvive.so" ] \
     && [ "$(git -C "$LIBSURVIVE_DIR" rev-parse HEAD 2>/dev/null)" = "$LIBSURVIVE_COMMIT" ]; then
    echo "[bootstrap] libsurvive already built at $LIBSURVIVE_DIR (${LIBSURVIVE_COMMIT:0:12})"
  else
    _missing=""
    for _pkg in libusb-1.0-0-dev liblapacke-dev libopenblas-dev cmake build-essential; do
      dpkg -s "$_pkg" >/dev/null 2>&1 || _missing="$_missing $_pkg"
    done
    if [ -n "$_missing" ]; then
      echo "[bootstrap] ERROR: libsurvive build deps missing:$_missing" >&2
      echo "[bootstrap]   sudo apt-get install -y$_missing     # then re-run ./bootstrap.sh" >&2
      exit 1
    fi
    if [ ! -d "$LIBSURVIVE_DIR/.git" ]; then
      echo "[bootstrap] cloning libsurvive -> $LIBSURVIVE_DIR"
      git clone https://github.com/cntools/libsurvive.git "$LIBSURVIVE_DIR"
    fi
    git -C "$LIBSURVIVE_DIR" cat-file -e "$LIBSURVIVE_COMMIT^{commit}" 2>/dev/null \
      || git -C "$LIBSURVIVE_DIR" fetch --quiet origin
    echo "[bootstrap] building libsurvive @ ${LIBSURVIVE_COMMIT:0:12} (the pinned commit; system libusb)"
    git -C "$LIBSURVIVE_DIR" checkout --quiet "$LIBSURVIVE_COMMIT"
    git -C "$LIBSURVIVE_DIR" submodule update --init --recursive --quiet
    ( cd "$LIBSURVIVE_DIR" && make -j"$(nproc)" )
    [ -f "$LIBSURVIVE_DIR/bin/libsurvive.so" ] \
      || { echo "[bootstrap] ERROR: the build produced no $LIBSURVIVE_DIR/bin/libsurvive.so" >&2; exit 1; }
  fi
  # The exact env the launcher gives the tracker process; prove the import + the two
  # symbols the bridge uses (SimpleContext, simple_serial_number) resolve from it.
  LD_LIBRARY_PATH="$LIBSURVIVE_DIR/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  PYTHONPATH="$LIBSURVIVE_DIR/bindings/python${PYTHONPATH:+:$PYTHONPATH}" \
    "$VIVE_PY" -c "import pysurvive; assert hasattr(pysurvive, 'SimpleContext') and hasattr(pysurvive, 'simple_serial_number'); print('[bootstrap] pysurvive OK from', pysurvive.__file__)"
else
  echo "[bootstrap] --viz-only: skipping the py3.9 camera venv + the libsurvive build"
fi

# 3) Seed the writable per-rig payload from the template (only if absent).
#    Recalibration ("Recompensation") overwrites this file in place.
COMP_DIR="$ROOT/robot_assets/bota_config/bota_payload_utils"
if [ ! -f "$COMP_DIR/payload_compensator.json" ]; then
  cp "$COMP_DIR/payload_compensator.example.json" "$COMP_DIR/payload_compensator.json"
  chmod u+w "$COMP_DIR/payload_compensator.json"
  echo "[bootstrap] seeded payload_compensator.json from the example template"
fi

cat <<EOF

[bootstrap] done.
  UI-only smoke (no hardware):
    PYTHONPATH=third_party/humi_ik .venv/bin/python viz/collection_viz.py --mock
    # open http://localhost:8080
  Full stack:
    ./run_collection_viz.sh
  Generate the per-rig Bota payload (after mounting your tool):
    .venv/bin/python force_torque/estimate_payload.py --max-time 60
EOF
