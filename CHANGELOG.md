# Changelog

## [Unreleased] - 2026-11-02

### Features
- `third_party/UMI_collection/`: the data collection software of the robot-free interface, which
  `scripts/collect_data.sh` wraps. Vive Tracker, RealSense D405, Bota F/T and DM4310 gripper bridges;
  the live viser scene with the pre-launch go/no-go status board; the out-of-process recorder; the
  session audit and per-take QC sheets; the converter from a raw session to the UMI Zarr (and LeRobot)
  dataset that `flore/` trains on; the raw / Zarr / LeRobot replay viewers; the G1 + YAM URDF, Bota
  configurations and rig calibration under `robot_assets/`.
- README: a "Collecting data" quick start; THIRD_PARTY_NOTICES: HuMI, i2rt, imagecodecs and Bota
  Systems entries for the vendored code, meshes and configurations.

### Design Rationale
- Vendored as plain files, like `third_party/holosoma_extensions`, rather than as a submodule:
  `git clone` alone yields a working collection stack, and the flore code that reads the G1 + YAM
  URDF (`third_party/UMI_collection/robot_assets/g1_yam/`) finds it without extra steps.
- Only what collection, conversion, QC and replay need is included. The motion-completion
  integration (its model and checkpoint are unreleased) and the real-robot visualizer are left out;
  the Zarr viewer's `--motion-completion` panel reports itself unavailable.
- The directory keeps its own environments (`bootstrap.sh`: a py3.12 hub venv, a py3.9 venv for the
  cp39-only RealSense wheel, libsurvive built from source) and its own ruff exemptions
  (`third_party/UMI_collection/ruff.toml`, style codes only), so its code stays identical to the
  rig's copy.

### Notes & Caveats
- With `third_party/UMI_collection/` present, the root test suite runs the URDF-backed tests; the
  ones that need `yourdfpy` skip unless the `viz` extra is installed.
- Device serials, ports and calibration files are those of the authors' rig and can be overridden
  through environment variables (see `third_party/UMI_collection/README.md`).
- The right-hand gripper keys are `j` = open / `k` = close (the left hand: `f` = close / `g` = open);
  the `scripts/collect_data.sh` help text is corrected accordingly.
