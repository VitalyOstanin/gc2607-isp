# TODO

## Contents

- [Image quality](#image-quality)
- [Performance](#performance)

## Image quality

- [ ] Finish the low-light denoise verification. Assess the gain-adaptive
      chroma (spatial) and luma (temporal) denoise across the gain range on real
      dim scenes: confirm it removes the expected noise, leaves detail intact,
      and shows no ghosting on motion; adjust the per-gain strength tables
      (`chroma_denoise_for_gain`, `temporal_luma_for_gain` in
      `src/bin/video.rs`) if needed. Pair this with the chroma box-blur
      running-sum optimization.

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
