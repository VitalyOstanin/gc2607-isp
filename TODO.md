# TODO

## Contents

- [Image quality](#image-quality)
- [Configuration](#configuration)
- [Performance](#performance)
- [Tooling](#tooling)
- [CI](#ci)

## Image quality

- [ ] Finish the low-light denoise verification. Assess the gain-adaptive
      chroma (spatial) and luma (temporal) denoise across the gain range on real
      dim scenes: confirm it removes the expected noise, leaves detail intact,
      and shows no ghosting on motion; adjust the per-gain strength tables
      (`chroma_denoise_for_gain`, `temporal_luma_for_gain` in
      `src/bin/video.rs`) if needed. Pair this with the chroma box-blur
      running-sum optimization.

## Configuration

- [ ] Let the chroma-denoise radius be reachable at run time, as a key of
      the run-time tuning file (`src/colour.rs`). `--denoise <scale>`
      multiplies only the blend strength; the box-blur radius comes from the
      gain-index table in `chroma_denoise_for_gain` (`src/bin/video.rs`), and at
      gain indices 0..3 the radius is 0, so in good light chroma speckle cannot
      be attenuated at all without a rebuild. Either let the scaler also raise
      the radius, or expose radius and the gain thresholds as tuning keys.

## Performance

- [ ] GPU backend: overlap CPU and GPU per frame (deferred — needs measurement
      and test coverage before implementing). Today `GpuProcessor::process`
      (`src/gpu.rs`) is fully serial: `submit` -> `map_async` ->
      `poll(Wait, timeout=None)` -> readback, on a single set of
      `raw_buf`/`yuyv_buf`/`staging_buf`. CPU idles in `poll` while the GPU runs
      and vice versa. Pipelining (double/triple buffering + deferred readback so
      `process(frame N)` returns frame N-1) would overlap them, but it changes
      the frame the AE loop measures and interacts with the `APPLY_DELAY` gain
      delay in `src/bin/video.rs`; the single-frame golden GPU test cannot catch
      a frame-ordering bug. Prerequisites: (1) add per-frame timing
      (CPU/GPU/poll) to quantify the serialization cost — at 30 fps the target is
      already met, so the benefit is latency/headroom, not throughput; (2) if
      worthwhile, implement a ring buffer with first-frame priming, re-tune the
      AE apply-delay, and add a multi-frame ordering test.

## Tooling

- [ ] Replace the hand-rolled CLI argument parsing with an established crate
      (`clap`, derive API). Each binary (`src/bin/video.rs`, `src/bin/capture.rs`,
      `src/bin/gpu_probe.rs`) currently parses `std::env::args()` by hand with a
      `match` loop and silent `unwrap_or` fallbacks on bad values; there is no
      shared parser, the flag set keeps growing, and `--help`/usage strings are
      maintained manually. A derive-based parser would give validation, generated
      help, and a single source of truth for the shared flags.

## CI

- [ ] Pin the CI runner host to the current Ubuntu LTS instead of `ubuntu-latest`.
      GitHub keeps `ubuntu-latest` on the previous LTS for months after a new LTS
      ships (until its runner image leaves preview), so `ubuntu-latest` is not the
      latest LTS. `build-deb.yml` already builds the package inside an
      `ubuntu:26.04` container (for libcamera 0.7), but the runner host
      (`runs-on: ubuntu-latest`) still trails; pin it to the matching LTS runner
      image (`ubuntu-26.04`, preview as of 2026-06) and bump when a newer LTS ships.
