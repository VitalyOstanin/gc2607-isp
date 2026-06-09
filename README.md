# gc2607-isp

*Read this in Russian: [README.ru.md](README.ru.md).*

> **Status: testing / experimental.** Targets one specific device (MateBook X
> Pro 2024 GC2607); it works end-to-end on the development machine, but
> interfaces, defaults and tuning may still change.

A software ISP (image signal processor) in Rust for the built-in colour camera
GalaxyCore GC2607 of the Huawei MateBook X Pro 2024 (Intel Meteor Lake, IPU6)
on Linux. It turns the sensor's raw Bayer frame into a colour image using
extracted sensor tuning.

**Image quality is the primary goal**: the point of this project is a better
picture than the generic software ISP gives (see [Features](#features) and
[Purpose](#purpose)).

The laptop has two GalaxyCore sensors. This project drives only the front
**colour** camera, **GC2607** (ACPI `GCTI2607`). The other one, **GC1029**
(ACPI `GCTI1029`), is a separate **infra-red** sensor (Windows Hello) on a
different I2C bus; it is **not needed** for a colour webcam, has no Linux driver
here, and is left alone.

## Contents

- [Purpose](#purpose)
- [Features](#features)
- [Architecture: the camera stack](#architecture-the-camera-stack)
- [Processing pipeline](#processing-pipeline)
- [Tuning source](#tuning-source)
- [Layout](#layout)
- [Installation (Ubuntu 26.04)](#installation-ubuntu-2604)
  - [Prebuilt packages](#prebuilt-packages)
  - [From source](#from-source)
- [Build and run (offline CLI)](#build-and-run-offline-cli)
- [Live webcam (gc2607-video)](#live-webcam-gc2607-video)
  - [Troubleshooting: washed-out / banded image after changing resolution](#troubleshooting-washed-out--banded-image-after-changing-resolution)
- [Building on other distributions](#building-on-other-distributions)
- [Debian package (.deb)](#debian-package-deb)
- [Correctness check (golden)](#correctness-check-golden)
- [Roadmap](#roadmap)
- [Related projects](#related-projects)
- [Acknowledgements](#acknowledgements)

## Purpose

The libcamera SoftISP produces a usable but mediocre image: simple debayer,
residual grey-world AWB tint, no vignetting correction. This project implements
its own CPU pipeline using extracted sensor tuning (colour-correction
matrices, lens-shading maps, white locus) that the libcamera tuning does not
have.

## Features

The main goal is image quality; everything below serves that, with performance
and power kept in check so it is usable as a daily webcam on a thin laptop.

**Image processing** (per-device factory tuning, see [Tuning source](#tuning-source)):

- black-level subtraction and per-Bayer-channel **lens-shading correction** with
  scene-based light-source selection (strong vignetting, gain up to ~5.5x);
- **robust-neutral auto white balance** (median chroma of bright, non-clipped
  pixels) with colour-temperature estimation against the sensor's white locus;
- **full-resolution Malvar-He-Cutler debayer** (own row-parallel implementation),
  plus a lighter half-res mode;
- **hue-sectored colour correction**: 5 CCMs interpolated by CCT, refined by 24
  per-hue-sector matrices fading to the global matrix near neutral;
- **lateral chromatic-aberration correction**;
- **gain-adaptive denoise**: chroma box blur + temporal luma (IIR with a motion
  gate), engaged only as analogue gain rises in dim light;
- sRGB gamma output, packed YUYV 4:2:2.

**Auto-exposure**: an exposure-priority loop drives exposure/gain/vblank directly
on the sensor over the V4L2 subdev, metering the produced frame's mean luminance
in linear light (no dependence on libcamera's AGC).

**GPU backend**: the per-pixel stages run as Vulkan compute shaders (wgpu, Mesa
ANV on the Intel iGPU), validated to within 1 LSB of the CPU path; `--backend
auto` prefers the GPU and cuts whole-process CPU use ~4x at the same frame rate.

**On-demand capture**: the camera is opened and the ISP runs only while an
application is capturing (v4l2loopback client-usage event); the sensor stays
runtime-suspended while idle.

**Integration**: the GC2607 is picked by its libcamera `Model` name, not by
enumeration index, so it is found correctly even with another camera (e.g. a
plugged-in USB webcam) present. The output is a v4l2loopback node any webcam
application opens (browser, OBS, `ffplay`); the privacy LED lights automatically
while streaming (handled by the V4L2 core / INT3472), and the laptop's physical
side camera switch is respected (with it off, the sensor delivers no usable
frame). An offline CLI converts a saved raw frame to PNG/PPM on any machine.

## Architecture: the camera stack

This ISP is one stage in a chain from the raw sensor to a regular webcam node.
Each layer below feeds the next:

```
GC2607 sensor (MIPI CSI-2, I2C control)
  -> gc2607.ko              V4L2 subdev driver: streaming + exposure/gain/vblank controls, runtime PM
  -> ipu-bridge (patched)   binds the GCTI2607 sensor to the Intel IPU6 (ACPI/_DSM clock + bridge table)
  -> intel_ipu6 / ipu6-isys IPU6 CSI-2 receiver: delivers the raw Bayer frame, exposes V4L2 capture + subdev nodes
  -> libcamera (0.7)        enumerates the IPU6 pipeline + the gc2607 sensor; used here ONLY to pull the raw stream
  -> gc2607-video (this)    the ISP: raw Bayer -> colour YUYV (CPU or GPU); AE drives the sensor over the V4L2 subdev
  -> v4l2loopback           this binary is the producer; the loopback node looks like a normal webcam
  -> application            browser / OBS / ffplay opens /dev/videoN
```

Two things deserve explanation: why the colour processing is a separate binary
rather than libcamera's own ISP, and why frames reach applications through
v4l2loopback rather than a GStreamer element.

### Why a custom ISP, not libcamera's SoftISP

libcamera ships a software ISP (the "simple" / soft pipeline) for sensors without
a hardware ISP. It is intentionally minimal: basic debayer, grey-world AWB,
gamma, and it only exposes Gamma/Contrast/Saturation controls. It cannot apply
the per-device factory tuning this project relies on for image quality:

- 5 colour-correction matrices interpolated by colour temperature, plus 24
  per-hue-sector matrices (see [Processing pipeline](#processing-pipeline) and
  [Tuning source](#tuning-source));
- per-Bayer-channel lens-shading correction with light-source selection;
- lateral chromatic-aberration correction;
- gain-dependent chroma/luma denoise;
- a GPU (Vulkan) backend for the per-pixel stages.

It also has no manual-exposure path: in raw mode libcamera's IPA does not run and
request controls are dropped, so a custom auto-exposure loop cannot go through it.
This project therefore uses libcamera **only as a raw-capture API** and controls
exposure/gain/vblank directly over the V4L2 subdev ([src/sensor.rs](src/sensor.rs)),
which runs in parallel with libcamera's raw stream.

### Why not integrate the ISP into libcamera or GStreamer

The processing could in principle live inside libcamera (extending the soft IPA)
or inside a GStreamer element. Neither is done, on purpose:

- **Inside libcamera** would mean forking the SimplePipelineHandler and its soft
  IPA (C++) and tracking their internal API across releases. The IPA interface is
  not meant to host an out-of-tree pipeline with a wgpu/Vulkan backend; the
  maintenance cost is high and it ties the project to one libcamera version.
- **As a GStreamer element** would mean either using `libcamerasrc` (which routes
  through the soft ISP rejected above) or writing and shipping a custom GStreamer
  plugin wrapping the Rust ISP. That is extra surface for no gain here: the binary
  already produces finished YUYV frames, so it writes them straight to a
  v4l2loopback node, which every webcam application opens with no plugin.

### Relationship to v4l2-relayd

[v4l2-relayd](https://github.com/9elements/v4l2-relayd) is the Ubuntu/IPU6 helper
that feeds a v4l2loopback node on demand from a GStreamer source. It is not used
as the producer here for the same reason: its source is a GStreamer pipeline,
which would bypass this ISP. Its **on-demand mechanism**, however — the
v4l2loopback `V4L2_EVENT_PRI_CLIENT_USAGE` event, which fires when an application
starts or stops capturing — is the right way to power the camera only while it is
in use, and is adopted directly inside `gc2607-video` (the `--on-demand` flag)
rather than by running relayd (see [Roadmap](#roadmap)).

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
| [packaging/](packaging/) | systemd unit + loopback config to run the daemon as an on-demand service |

## Installation (Ubuntu 26.04)

The colour webcam needs all three parts of the stack
([Architecture](#architecture-the-camera-stack)): the `gc2607` sensor driver,
the patched `ipu-bridge`, and this ISP. Each is published as a Debian package
on its GitHub releases page. The fastest route on Ubuntu 26.04 is to install the
three prebuilt packages; building everything from source is also supported
([From source](#from-source)).

This targets the MateBook X Pro 2024 specifically and is verified on Ubuntu
26.04 with a 7.x kernel.

### Prebuilt packages

Prerequisites:

- Ubuntu 26.04 with a 7.x kernel and the matching kernel headers
  (`linux-headers-$(uname -r)`), which DKMS needs to build the modules.
- The physical camera switch on the side of the laptop must be **on**, or the
  sensor delivers no usable frame.
- With Secure Boot enabled, DKMS signs the modules with a local MOK key and
  prompts to enrol it on first build; complete the enrolment (a reboot asks to
  confirm the key) so the signed modules load.

1. Download the three packages from their latest releases (uses the GitHub CLI;
   alternatively download them from each project's releases page in a browser):

   ```sh
   mkdir gc2607-debs && cd gc2607-debs
   gh release download v1.0   -R VitalyOstanin/gc2607-driver
   gh release download v1.0   -R VitalyOstanin/gc2607-ipu-bridge
   gh release download v0.1.0 -R VitalyOstanin/gc2607-isp
   ```

2. Install all three at once so `apt` resolves the dependencies and the
   recommended packages (`v4l2loopback-dkms`, `mesa-vulkan-drivers`) together:

   ```sh
   sudo apt install ./gc2607-driver-dkms_1.0_all.deb \
                    ./gc2607-ipu-bridge-dkms_1.0_all.deb \
                    ./gc2607-isp_0.1.0_amd64.deb
   ```

   The two DKMS packages build and sign their kernel modules for the running 7.x
   kernel (other kernel series are skipped via `BUILD_EXCLUSIVE_KERNEL`). The
   `gc2607-isp` package installs `gc2607-video` to `/usr/bin`, the loopback
   configuration, and the `gc2607-camera` service (enabled for the next boot but
   not started during install).

3. Reboot so the new `ipu-bridge` replaces the in-tree one and the IPU6 builds
   the CSI-2 link to the sensor. After the reboot the loopback module is loaded
   from `/etc/modules-load.d/`, the `gc2607-camera` service is running, and the
   camera stays idle until an application opens it.

   ```sh
   sudo reboot
   ```

4. Verify:

   ```sh
   systemctl status gc2607-camera        # active (running), sensor idle
   v4l2-ctl --list-devices               # lists "MateBook Camera (GC2607)"
   ffplay -f v4l2 /dev/video0            # opening the node wakes the camera
   ```

   Any webcam application (browser, OBS, `ffplay`) can now open
   "MateBook Camera (GC2607)" as a regular webcam. The sensor powers up only
   while an application is capturing and suspends again when it stops.

To remove the webcam service and loopback configuration, see
[`packaging/README.md`](packaging/README.md#uninstall); the two DKMS packages are
removed with `sudo apt remove gc2607-driver-dkms gc2607-ipu-bridge-dkms`.

### From source

To build the stack from source instead of installing the prebuilt packages:

- **Offline CLI** (`gc2607-isp`, raw frame → image, no hardware needed) —
  [Build and run (offline CLI)](#build-and-run-offline-cli).
- **Live webcam binary** (`gc2607-video`) — [Live webcam](#live-webcam-gc2607-video)
  for the container build, or [Building on other distributions](#building-on-other-distributions)
  for a native build.
- **Debian package** for this ISP — [Debian package (.deb)](#debian-package-deb).
- **Sensor driver and ipu-bridge** are built from their own repositories
  ([gc2607-driver](https://github.com/VitalyOstanin/gc2607-driver),
  [gc2607-ipu-bridge](https://github.com/VitalyOstanin/gc2607-ipu-bridge));
  each ships a DKMS package built with `dpkg-buildpackage -us -uc -b`.

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
   sudo modprobe v4l2loopback card_label="MateBook Camera (GC2607)" exclusive_caps=1
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

Then open the loopback node (`/dev/video0`, e.g. "MateBook Camera (GC2607)") in
any app, or preview it directly:

```sh
ffplay -f v4l2 /dev/video0
```

### Options

| Flag | Default | Meaning |
|------|---------|---------|
| `--device /dev/videoN` | `/dev/video0` | v4l2loopback output node |
| `--device-label <name>` | (unset) | locate the loopback by its `card_label` instead of a fixed `/dev/videoN`; overrides `--device`. Robust to probe order (the IPU6 ISYS registers dozens of `/dev/video*`). Used by the systemd service. |
| `--backend auto\|cpu\|gpu` | `auto` | `auto` prefers the GPU and falls back to the CPU if none is available. `gpu` forces the GPU (per-pixel stages as Vulkan compute, AWB on CPU; always full-res MHC, ignores `--debayer`) and fails if it cannot initialise. `cpu` forces the CPU path. GPU needs the `gpu` build feature; without it `auto` is CPU and `gpu` errors. |
| `--debayer half\|mhc` | `mhc` | `mhc` = full-res 1920x1080; `half` = 960x540, lighter (CPU backend only) |
| `--threads N` | `8` | ISP worker threads (CPU budget; see note). On the GPU backend only the occasional CPU-side AWB uses them. |
| `--no-ae` | (AE on) | disable auto-exposure (use current sensor settings) |
| `--target <0..1>` | `0.15` | AE target mean brightness, in linear light (the output luma is metered with the sRGB gamma inverted). `0.15` linear ≈ gamma-encoded mid-grey `0.43`. |
| `--max-gain <idx>` | `16` | AE max analogue-gain LUT index |
| `--on-demand auto\|on\|off` | `auto` | open the camera and run the ISP only while an application is capturing (see [Roadmap](#roadmap)). `auto` self-gates when v4l2loopback exposes the client-usage event and otherwise streams always-on; `on` requires the event (warns and falls back if absent); `off` forces always-on. While idle the camera sensor stays runtime-suspended; the producer only writes cheap black keepalive frames so consumers can still negotiate the format. |

Performance (Core Ultra 185H): `mhc` sustains the sensor's 30 fps at 8 threads
(~29 at 4); `half` reaches 30 fps at 4 threads (~27 single-threaded). The
default (`mhc`, 8 threads) maximises quality; lower `--threads` or use
`--debayer half` to reduce CPU load and heat on a thermally constrained laptop.

The GPU backend produces the same full-res MHC output (validated against the CPU
path to within 1 LSB, `tests/gpu.rs`) while offloading the per-pixel work. In one
fixed scene the whole-process CPU use dropped from ~154% (CPU `mhc`, 8 threads)
to ~37% (GPU) at the same frame rate — roughly a 4x CPU reduction, which matters
on this thermally constrained laptop.

### Troubleshooting: washed-out / banded image after changing resolution

v4l2loopback keeps the capture-side format that was first fixed on the node. If
the daemon is run once at half-res (`--debayer half`, 960x540) and then at the
default full-res (`mhc`, 1920x1080) *without reloading the module*, the producer
writes 1920x1080 frames into a node whose capture format is still pinned to
960x540. Consumers then dequeue short frames (e.g. `1026048` bytes instead of
the expected size), which shows up as a low-contrast, washed-out, colour-shifted
image with horizontal banding (a torn frame), not as an outright error.

Fix: reload the module so the format is reset, then start the daemon at the
intended resolution:

```sh
sudo modprobe -r v4l2loopback && sudo modprobe v4l2loopback
```

To avoid it, run the daemon at a consistent resolution (the default full-res
`mhc` is recommended). Switching `--debayer` between runs requires a module
reload.

### Run as a system service (on-demand)

[`packaging/`](packaging/) holds a systemd unit and the loopback configuration to
run `gc2607-video` as an on-demand service: the virtual webcam is always present,
but the sensor and ISP run only while an application captures. The loopback is
created at boot with a fixed `card_label` and the daemon finds it by label
(`--device-label`), so no `/dev/videoN` number is hard-coded. Telemetry (stdout)
is discarded; warnings and errors (stderr) go to the journal.

Install and management steps, plus a hardened (non-root) variant, are in
[`packaging/README.md`](packaging/README.md). In short:

```sh
sudo install -m 0755 gc2607-video /usr/bin/gc2607-video
sudo install -m 0644 packaging/modules-load.d/gc2607-loopback.conf /etc/modules-load.d/
sudo install -m 0644 packaging/modprobe.d/gc2607-loopback.conf     /etc/modprobe.d/
sudo install -m 0644 packaging/systemd/gc2607-camera.service       /etc/systemd/system/
sudo modprobe v4l2loopback
sudo systemctl daemon-reload && sudo systemctl enable --now gc2607-camera
```

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

## Debian package (.deb)

For Ubuntu 26.04 the live webcam is packaged as `gc2607-isp` (amd64): it installs
`gc2607-video` to `/usr/bin`, the on-demand systemd service and the loopback
configuration (the files in [`packaging/`](packaging/)). The binary is built
separately and the package only wraps it (`debian/rules` overrides the compile
steps), so `dpkg-shlibdeps` derives the `libcamera0.7` dependency by scanning the
ELF.

Build it after the binary exists:

```sh
# build the binary in the container (see "Live webcam" above), then copy it to the
# repository root where debian/install expects it:
cp target/release/gc2607-video ./gc2607-video
sudo apt install -y debhelper dpkg-dev          # packaging tools, once
dpkg-buildpackage -us -uc -b                     # -> ../gc2607-isp_0.1.0_amd64.deb
sudo apt install ../gc2607-isp_0.1.0_amd64.deb
```

The package `Recommends` `mesa-vulkan-drivers`, `v4l2loopback-dkms`,
`gc2607-driver-dkms` and `gc2607-ipu-bridge-dkms`. It enables `gc2607-camera.service`
for the next boot but does not start it during install (the loopback node is
created at boot); see [`packaging/README.md`](packaging/README.md).

### CI

[`.github/workflows/build-deb.yml`](.github/workflows/build-deb.yml) builds the
binary and the `.deb` inside an `ubuntu:26.04` container (to match libcamera 0.7)
on every push and pull request and uploads the `.deb` as a workflow artifact. On a
`v*` tag it also attaches the `.deb` to a GitHub release. The GPU feature is
compiled but not runtime-tested (the hosted runners have no GPU).

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
| 4. AE loop | done | auto-exposure, exposure-priority, linear-light metric, hardware-validated |
| 5. Output | done | write frames to v4l2loopback `/dev/video0` |
| 6. Performance | done | `Processor` (buffer reuse, AWB/grid caching), parallel front-end + MHC + YUYV pack; 30 fps at full-res |
| 7. GPU offload | done | `--backend gpu`: per-pixel stages (unpack/BLC/LSC/WB, MHC, CCM, gamma, YUYV pack) as Vulkan compute shaders; AWB on the CPU. ~4x lower CPU at the same frame rate |
| 8. On-demand capture | done | `--on-demand auto` opens the camera + runs the ISP only while an application is capturing, driven by the v4l2loopback `V4L2_EVENT_PRI_CLIENT_USAGE` event (the v4l2-relayd mechanism, implemented in-process); the sensor stays runtime-suspended while idle. Falls back to always-on where the event is unavailable |

## Related projects

This ISP is the last stage of a three-part stack (see
[Architecture](#architecture-the-camera-stack)):

- [gc2607-driver](https://github.com/VitalyOstanin/gc2607-driver) — the GC2607 V4L2 sensor driver (I2C bring-up, exposure/gain/vblank, runtime PM).
- [gc2607-ipu-bridge](https://github.com/VitalyOstanin/gc2607-ipu-bridge) — the IPU-bridge patch that registers the `GCTI2607` HID with the IPU6 so the CSI-2 link is built.

## Acknowledgements

Two earlier community efforts to run the MateBook GC2607 on Linux were the
starting point that got this work going — thanks to their authors:

- [abbood/gc2607-v4l2-driver](https://github.com/abbood/gc2607-v4l2-driver) — a
  GC2607 V4L2 sub-device driver with INT3472 power and an `ipu_bridge` patch,
  feeding a `v4l2loopback` node via GStreamer (developed on Arch). The basis for
  this project's own [gc2607-driver](https://github.com/VitalyOstanin/gc2607-driver).
- [antonbiluta/gc2607-driver](https://github.com/antonbiluta/gc2607-driver) — a
  GC2607 driver and `ipu_bridge` patch (Fedora), another early reference for
  bringing the sensor up.

This project then went its own way — a from-scratch sensor driver written to the
kernel camera-sensor guidelines, a dedicated `ipu-bridge` patch, and this Rust
ISP with extracted factory tuning — but those projects showed it was possible
and where to start.
