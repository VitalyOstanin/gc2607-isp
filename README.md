# gc2607-isp

A software ISP (image signal processor) in Rust for the built-in colour camera
GalaxyCore GC2607 of the Huawei MateBook X Pro 2024 (Intel Meteor Lake, IPU6)
on Linux. It turns the sensor's raw Bayer frame into a colour image using
extracted sensor tuning.

## Contents

- [Purpose](#purpose)
- [Processing pipeline](#processing-pipeline)
- [Tuning source](#tuning-source)
- [Layout](#layout)
- [Build and run (offline CLI)](#build-and-run-offline-cli)
- [Live webcam (gc2607-video)](#live-webcam-gc2607-video)
- [Building on other distributions](#building-on-other-distributions)
- [Correctness check (golden)](#correctness-check-golden)
- [Roadmap](#roadmap)

## Purpose

The libcamera SoftISP produces a usable but mediocre image: simple debayer,
residual grey-world AWB tint, no vignetting correction. This project implements
its own CPU pipeline using extracted sensor tuning (colour-correction
matrices, lens-shading maps, white locus) that the libcamera tuning does not
have.

## Processing pipeline

```
raw SGRBG10 (1928x1088)
  -> BLC          black-level subtraction (pedestal 64)
  -> LSC          per-Bayer-channel vignetting correction (light source by scene)
  -> AWB          robust-neutral: median chroma of bright, non-clipped pixels
  -> debayer      half-res (stage 1); full-res MHC (stage 2)
  -> CCM          hue-sectored colour correction (24 sectors), interpolated by CCT
  -> sRGB gamma
  -> RGB8
```

White balance, CCT and the lens-shading light source are estimated on the
post-BLC frame, before LSC is applied.

The colour step is hue-sectored: beyond the single global matrix, 24 per-hue
matrices refine the correction by the colour's hue, fading to the global matrix
near neutral. See [docs/acm-color-model.md](docs/acm-color-model.md).

## Tuning source

Tuning is parsed from the camera's `.aiqb` tuning file:

- **CCM** — 5 colour-correction matrices by CCT (2963/3803/4049/4779/6426 K),
  plus a white locus for CCT estimation. File [data/gc2607_ccms.json](data/gc2607_ccms.json).
- **ACM** — 24 per-hue-sector colour matrices for each of the 5 CCTs (hue-sectored
  refinement of the CCM). File [data/gc2607_acm.npz](data/gc2607_acm.npz); see
  [docs/acm-color-model.md](docs/acm-color-model.md).
- **LSC** — 5 light sources x 4 Bayer channels x 63x47 grid, gain 1.0..5.543
  (strong vignetting). File [data/gc2607_lsc.npz](data/gc2607_lsc.npz).
- **LCA** — lateral chromatic aberration shift grids (red/blue vs green).
  File [data/gc2607_lca.npz](data/gc2607_lca.npz).

The tone curve is not extractable (parsed by Intel's closed parser), so a
standard sRGB gamma is used.

Rust data is generated from the sources by [tools/gen_tuning.py](tools/gen_tuning.py)
(produces [src/tuning_data.rs](src/tuning_data.rs) and `data/lsc_grids.bin`).

## Layout

| Path | Purpose |
|------|---------|
| [src/raw.rs](src/raw.rs) | raw SGRBG10 loading + black-level subtraction |
| [src/pipeline.rs](src/pipeline.rs) | core: debayer (half / own MHC), AWB, CCT, LSC, hue-sectored CCM, gamma; `Processor` (live, buffer-reuse + caching) |
| [src/tuning.rs](src/tuning.rs) | access to the embedded LSC / LCA grids |
| [src/tuning_data.rs](src/tuning_data.rs) | generated tables (CCM, ACM sectors, locus, dims) |
| [src/ae.rs](src/ae.rs) | auto-exposure logic (exposure-priority), std-only, unit-tested |
| [src/sensor.rs](src/sensor.rs) | V4L2 subdev control (exposure/gain/vblank) via raw ioctls (`video` feature) |
| [src/output.rs](src/output.rs) | v4l2loopback output, RGB->YUYV pack (`video` feature) |
| [src/gpu.rs](src/gpu.rs) | GPU backend: `GpuProcessor` + WGSL compute shaders (unpack/LSC/WB, MHC, hue-sectored CCM, gamma, YUYV) (`gpu` feature) |
| [src/main.rs](src/main.rs) | offline CLI: raw -> PNG/PPM |
| [src/bin/capture.rs](src/bin/capture.rs) | `gc2607-capture`: grab raw frames via libcamera (`capture` feature) |
| [src/bin/video.rs](src/bin/video.rs) | `gc2607-video`: live webcam daemon (`capture` feature; `--backend gpu` adds `gpu`) |
| [src/bin/gpu_probe.rs](src/bin/gpu_probe.rs) | `gpu-probe`: minimal Vulkan compute sanity check (`gpu` feature) |
| [tools/reference_pipeline.py](tools/reference_pipeline.py) | reference pipeline (Python) |
| [tools/gen_golden.py](tools/gen_golden.py) | golden artifact generation |
| [tests/golden.rs](tests/golden.rs) | checks the Rust output against the Python reference + reference MHC crate |
| [tests/gpu.rs](tests/gpu.rs) | checks the GPU backend matches the CPU MHC path within 1 LSB (`gpu`+`video`) |

## Build and run (offline CLI)

The offline CLI converts a saved raw frame to an image. It is dependency-light
and builds on the host (no libcamera needed):

```sh
cargo build --release
./target/release/gc2607-isp [--half|--mhc] <input.bin> [output.png|output.ppm]
```

`input.bin` is a raw SGRBG10 1928x1088 frame, stride 1952 u16/line.

- `--mhc` (default): full-resolution Malvar-He-Cutler debayer, 1928x1088 output.
- `--half`: half-resolution debayer, 964x544, matches the golden reference.

Output format is chosen by extension: `.png` (via the `image` crate) or PPM
(P6) otherwise.

## Live webcam (gc2607-video)

`gc2607-video` runs the pipeline live: it captures raw via libcamera, runs the
ISP, drives auto-exposure on the sensor, and publishes processed frames to a
v4l2loopback node that any application (browser, OBS, `ffplay`) can open as a
regular webcam.

### Prerequisites

1. The GC2607 sensor stack is loaded and working: `gc2607.ko` + the patched
   `ipu-bridge`, so `/dev/v4l-subdev*` includes a `gc2607` subdev and libcamera
   sees the camera. The physical camera switch must be **on**.
2. libcamera 0.7 runtime is installed.
3. A v4l2loopback device exists for the output, e.g.:
   ```sh
   sudo modprobe v4l2loopback card_label="Virtual Camera" exclusive_caps=1
   ```
   Note the created node (this project defaults to `/dev/video0`).

### Build

`gc2607-video` needs the `capture` feature (the `libcamera` crate + clang), so
it is built in a podman container (Ubuntu 26.04), not on the host. Build with
`capture,gpu` so the default `auto` backend can use the GPU (the `gpu` feature is
self-contained — wgpu/Vulkan, mesa ANV on the Intel iGPU — and needs no extra
runtime beyond a working Vulkan driver):

```sh
# build inside the container (image built once from Containerfile)
podman run --rm --http-proxy=false \
  -v "$PWD":/work \
  -v gc2607-cargo-registry:/root/.cargo/registry \
  -v gc2607-cargo-target:/work/target \
  localhost/gc2607-isp-build:latest \
  cargo build --release --features capture,gpu --bin gc2607-video

# copy the binary out of the container's target volume to the host
podman run --rm \
  -v "$PWD":/work -v gc2607-cargo-target:/work/target \
  localhost/gc2607-isp-build:latest \
  cp /work/target/release/gc2607-video /work/gc2607-video
```

(Build the image first if needed: `podman build -t gc2607-isp-build:latest -f Containerfile .`)

To build a CPU-only binary, drop the `gpu` feature (`--features capture`); then
`auto` resolves to the CPU path and `--backend gpu` reports an error.

#### Why a container

The container is only for the `capture` feature — the offline CLI and the GPU
backend are pure Rust and build directly on the host. The `libcamera` crate
needs three things **at build time** that this project deliberately keeps off
the host: the libcamera headers (`libcamera-dev`), `clang` (the `libcamera-sys`
crate generates its bindings from those headers with bindgen), and
`pkg-config`. The image carries them plus its own pinned Rust toolchain; the
cargo registry and target directory are mounted as named volumes, so crates are
not re-downloaded and nothing is written into the host's `~/.cargo`. (Rust
itself is not the reason for the container — the host already has cargo and uses
it for the offline and GPU builds.)

The image is `ubuntu:26.04` specifically because its libcamera (0.7) matches the
laptop's runtime `libcamera.so.0.7`: the binary links libcamera **dynamically**,
is copied out to the host, and runs there against the host's libcamera runtime
(no `-dev` package needed on the host). Building against a different libcamera
version would produce a binary that does not match the host runtime.

If you are willing to install `libcamera-dev` and `clang` on the build machine,
you can skip the container and build natively — see
[Building on other distributions](#building-on-other-distributions).

### Run

```sh
./gc2607-video                 # default: --backend auto (prefer GPU, fall back to CPU), full-res 1080p
./gc2607-video --backend cpu   # force the CPU path (--debayer mhc --threads 8 by default)
```

Then open the loopback node (`/dev/video0` "Virtual Camera") in any app, or
preview it directly:

```sh
ffplay -f v4l2 /dev/video0
```

### Options

| Flag | Default | Meaning |
|------|---------|---------|
| `--device /dev/videoN` | `/dev/video0` | v4l2loopback output node |
| `--backend auto\|cpu\|gpu` | `auto` | `auto` prefers the GPU and falls back to the CPU if none is available. `gpu` forces the GPU (per-pixel stages as Vulkan compute, AWB on CPU; always full-res MHC, ignores `--debayer`) and fails if it cannot initialise. `cpu` forces the CPU path. GPU needs the `gpu` build feature; without it `auto` is CPU and `gpu` errors. |
| `--debayer half\|mhc` | `mhc` | `mhc` = full-res 1920x1080; `half` = 960x540, lighter (CPU backend only) |
| `--threads N` | `8` | ISP worker threads (CPU budget; see note). On the GPU backend only the occasional CPU-side AWB uses them. |
| `--no-ae` | (AE on) | disable auto-exposure (use current sensor settings) |
| `--target <0..1>` | `0.35` | AE target mean brightness |
| `--max-gain <idx>` | `16` | AE max analogue-gain LUT index |

Performance (Core Ultra 185H): `mhc` sustains the sensor's 30 fps at 8 threads
(~29 at 4); `half` reaches 30 fps at 4 threads (~27 single-threaded). The
default (`mhc`, 8 threads) maximises quality; lower `--threads` or use
`--debayer half` to reduce CPU load and heat on a thermally constrained laptop.

The GPU backend produces the same full-res MHC output (validated against the CPU
path to within 1 LSB, `tests/gpu.rs`) while offloading the per-pixel work. In one
fixed scene the whole-process CPU use dropped from ~154% (CPU `mhc`, 8 threads)
to ~37% (GPU) at the same frame rate — roughly a 4x CPU reduction, which matters
on this thermally constrained laptop.

## Building on other distributions

The project has three build profiles with different system requirements:

| Profile                                       | Cargo features    | Build-time system deps                                       | Runtime deps                                                  |
|-----------------------------------------------|-------------------|--------------------------------------------------------------|---------------------------------------------------------------|
| Offline CLI (`gc2607-isp`)                    | none (default)    | Rust toolchain only                                          | none                                                          |
| + GPU backend                                 | `gpu`             | Rust toolchain only (wgpu is vendored)                       | a working Vulkan driver (Mesa ANV for the Intel iGPU)         |
| Live / capture (`gc2607-video`, `gc2607-capture`) | `capture[,gpu]` | libcamera headers, `clang` (for bindgen), `pkg-config`, a C++ toolchain | libcamera runtime + the GC2607 sensor stack (see Prerequisites) |

The offline CLI and the GPU backend are pure Rust and build on any distribution
with a current stable toolchain; they were developed and tested with Rust
1.88.0 (pinned in `Containerfile`). Only the `capture` feature needs system
libraries.

### libcamera version

The `libcamera` crate (0.7) binds to the system libcamera through `pkg-config`
and `bindgen`. Only the current release of each distribution is targeted, and
all of them ship libcamera new enough for the crate (upstream: "known to build
with libcamera v0.4.0 and up"). As of 2026-06-08, per repology.org:

| Distribution         | libcamera                | Native `capture` build      |
|----------------------|--------------------------|-----------------------------|
| Ubuntu 26.04 LTS     | 0.7.0                    | yes (verified)              |
| Debian 13 (trixie)   | 0.4.0 (backports 0.7.1)  | should work (untested here) |
| Fedora 43 / 44       | 0.5.2 / 0.7.x            | should work (untested here) |

"yes (verified)" marks the only combination this project has actually been
built and run on — the podman `ubuntu:26.04` image, which matches the laptop's
runtime libcamera 0.7.0 ABI. The other rows meet the crate's `>= 0.4.0`
requirement but are untested here; API differences between libcamera minor
versions may need small adjustments.

Install the build dependencies, then build natively instead of in the container:

```sh
# Debian / Ubuntu
sudo apt install libcamera-dev clang pkg-config build-essential
# Fedora
sudo dnf install libcamera-devel clang pkgconf-pkg-config gcc-c++

cargo build --release --features capture,gpu
```

### Hardware scope

This ISP targets one specific device — the MateBook X Pro 2024 GC2607 camera.
The offline CLI is portable (it processes a saved raw frame on any machine), but
`gc2607-video` / `gc2607-capture` are only useful on that laptop, where the
`gc2607` sensor driver, the patched `ipu-bridge`, and the tuning are present. On
other hardware the capture binaries build but find no `gc2607` camera to open.

## Correctness check (golden)

`cargo test` checks the Rust pipeline against the validated Python reference on a
fixed frame:

- `estimate_matches_reference` — gains/CCT/LSC source match within 1e-3 (AWB
  relies on median/percentile, sensitive to operation order);
- `render_matches_reference` — the deterministic render at given gains/CCM/LSC
  matches the reference byte-for-byte (1 LSB tolerance on a few pixels).

Sensor captures and the golden artifacts derived from them are private to the
user's scene and are **not** committed (see `.gitignore`); the tuning in `data/`
is sensor tuning data, not a capture, and is tracked. With no local data
the tests skip instead of failing. To run them, place a raw frame at
`tests/data/sample-raw.bin` and generate the reference:
`python3 tools/gen_golden.py`.

## Roadmap

| Stage | State | Content |
|-------|-------|---------|
| 1. CPU core (offline) | done | half-res debayer, golden check vs Python |
| 2. Quality + CLI | done | own row-parallel MHC debayer (replaced the `demosaic` crate), PNG output |
| 3. Capture | done | raw + exposure/gain controls via the `libcamera` crate (built in podman ubuntu:26.04) |
| 4. AE loop | done | auto-exposure, exposure-priority (target ~35%), hardware-validated |
| 5. Output | done | write to v4l2loopback `/dev/video0` (binary only, no systemd service) |
| 6. Performance | done | `Processor` (buffer reuse, AWB/grid caching), parallel front-end + MHC + YUYV pack; 30 fps at full-res |
| 7. GPU offload | done | `--backend gpu`: per-pixel stages (unpack/BLC/LSC/WB, MHC, CCM, gamma, YUYV pack) as Vulkan compute shaders; AWB on the CPU. ~4x lower CPU at the same frame rate |
