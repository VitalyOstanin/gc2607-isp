# TODO

## Contents

- [Image quality](#image-quality)

## Image quality

- [ ] Finish the low-light denoise verification. Assess the gain-adaptive
      chroma (spatial) and luma (temporal) denoise across the gain range on real
      dim scenes: confirm it removes the expected noise, leaves detail intact,
      and shows no ghosting on motion; adjust the per-gain strength tables
      (`chroma_denoise_for_gain`, `temporal_luma_for_gain` in
      `src/bin/video.rs`) if needed. Pair this with the chroma box-blur
      running-sum optimization.
