# CLAUDE.md — gc2607-isp

Guidance for AI agents working in this repository.

## Contents

- [Project](#project)
- [Language policy](#language-policy)
- [Privacy: never commit sensor captures](#privacy-never-commit-sensor-captures)
- [Architecture](#architecture)
- [Tuning data](#tuning-data)
- [Build, run, test](#build-run-test)
- [Golden test](#golden-test)
- [Dependency policy](#dependency-policy)
- [Conventions](#conventions)

## Project

A software ISP in Rust for the GalaxyCore GC2607 colour camera (Huawei MateBook
X Pro 2024, Intel Meteor Lake / IPU6) on Linux. It converts a raw Bayer frame
into colour using extracted sensor tuning. The libcamera
SoftISP already produces a usable image; this project targets higher quality
(lens-shading, robust AWB, extracted CCMs) that the libcamera tuning lacks.

## Language policy

- All repository artifacts are in **English**: code, comments, identifiers,
  README, CLI messages, test messages, commit messages, docs.
- **Communicate with the user in Russian** (the user's standing preference).

## Privacy: never commit sensor captures

Raw frames from the sensor capture the user's real scene. Neither the captures
nor anything derived from them may be committed:

- ignored via `.gitignore`: `tests/data/*` (except `.gitkeep`), `*.ppm`, `*.png`.
- `tests/data/sample-raw.bin`, `golden_render.bin`, `golden.png`, `golden.json`
  are **local only**.
- Tuning in `data/` (`gc2607_ccms.json`, `gc2607_lsc.npz`, `lsc_grids.bin`) is
  sensor tuning data, not a scene capture, and **is** tracked.

When adding test fixtures or examples, never introduce a scene image into git.

## Architecture

Pipeline (`src/pipeline.rs`):

```
raw SGRBG10 1928x1088
  -> BLC (pedestal 64)                         src/raw.rs
  -> LSC (per Bayer channel, source by scene)  uses data/lsc_grids.bin
  -> AWB robust-neutral (median chroma)
  -> debayer  (half-res now; MHC stage 2)
  -> CCM hue-sectored (24 sectors), interpolated by CCT   docs/acm-color-model.md
  -> sRGB gamma
  -> RGB8
```

WB gains, CCT and the LSC light source are estimated on the post-BLC frame,
before LSC. Estimation lives in `pipeline::estimate`; the deterministic render
(LSC -> debayer -> WB -> hue-sectored CCM -> gamma) in `pipeline::render`. Keep
these two separable — the golden test exercises them independently. The colour
step (`pipeline::acm_color`, mirrored in `gpu.rs` WGSL and the Python reference)
applies 24 per-hue-sector matrices, fading to the global CCM near neutral; see
[docs/acm-color-model.md](docs/acm-color-model.md).

## Tuning data

The tuning in `data/` is parsed from the camera's `.aiqb` tuning file and
regenerated into Rust by `tools/gen_tuning.py` (produces `src/tuning_data.rs`
and `data/lsc_grids.bin`). Do not hand-edit `src/tuning_data.rs` — it is
generated.
The Python reference `tools/reference_pipeline.py` is the numeric source of
truth the Rust code must match; it is self-contained (reads `data/`).

## Build, run, test

```sh
cargo build --release
./target/release/gc2607-isp <input.bin> [output.ppm]
cargo test                       # golden tests (skip if no local data)
python3 tools/gen_tuning.py      # regenerate Rust tuning from data/
python3 tools/gen_golden.py      # regenerate golden artifacts (needs local raw)
```

Builds are currently hermetic (std-only, no network). Keep stage-1 core
dependency-free; add crates only per the dependency policy below.

## Golden test

`tests/golden.rs` checks the Rust pipeline against the Python reference on a
fixed raw frame:

- `estimate_matches_reference` — gains/CCT/LSC source within 1e-3 relative
  (AWB uses median/percentile, sensitive to float order).
- `render_matches_reference` — deterministic render at golden gains/CCM/LSC
  matches `golden_render.bin` byte-for-byte (<= 1 LSB on a few pixels).

Both **skip** (not fail) when local sensor data is absent. If you change the
pipeline numerics, regenerate the reference and confirm the diff is intended.

## Dependency policy

Staged, agreed with the user. Stage 1 (core) is intentionally std-only so the
build is hermetic and the golden test validates arithmetic before any crate is
introduced.

| Need | Crate | Verdict |
|------|-------|---------|
| Parallelism | `rayon` | trusted, active, pure Rust |
| PNG / image IO | `image` or `png` | trusted, active, pure Rust |
| MHC debayer | `demosaic` | low adoption (verify correctness; keep a hand-written MHC as fallback); isolate behind a debayer trait |
| libcamera capture | `libcamera` 0.7 | only option for IPU6; build in podman `ubuntu:26.04` (matches host libcamera 0.7 ABI); needs libcamera-dev + clang, which must NOT be installed on the host |
| v4l2loopback output | own code via `nix` | prefer a small custom V4L2 output (open + VIDIOC_S_FMT + frames) over the `v4l` crate, which is stale (2023) with no documented output API |

Custom V4L2 is for **output only**. Capture stays on libcamera (IPU6 CSI/ISYS
setup is not worth reimplementing). Keep the half-res debayer as the
golden-checked test mode even after MHC lands.

## Conventions

- No emoji except check/cross marks in test checklists.
- Do not raise parallelism/resource limits (rayon thread pool, etc.) without
  asking the user.
- Mark unverified claims; prefer reading/running over guessing.
- Commit and push only on explicit user request; never add `Co-Authored-By`.
