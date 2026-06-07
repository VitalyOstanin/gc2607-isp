#!/usr/bin/env python3
"""Generate golden artifacts for the Rust golden test.

Outputs (in ../tests/data):
  golden.json        -- meta: dims, scene_chroma, gains, cct, lsc_ls, ccm
  golden_render.bin  -- raw RGB8 bytes (height*width*3), row-major, the exact
                        reference render the Rust pipeline must reproduce.
  golden.png         -- same render as PNG, for human inspection.
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import reference_pipeline as rp  # type: ignore  # noqa: E402

TDATA = os.path.join(HERE, "..", "tests", "data")


def main():
    raw = os.path.join(TDATA, "sample-raw.bin")
    rgb8, meta = rp.process(raw)
    h, w, _ = rgb8.shape

    rgb8.tofile(os.path.join(TDATA, "golden_render.bin"))
    json.dump(meta, open(os.path.join(TDATA, "golden.json"), "w"), indent=2)

    ppm = "/tmp/_golden.ppm"
    with open(ppm, "wb") as f:
        f.write(f"P6\n{w} {h}\n255\n".encode())
        f.write(rgb8.tobytes())
    subprocess.run(["convert", ppm, os.path.join(TDATA, "golden.png")], check=True)
    os.remove(ppm)

    print(f"golden: {w}x{h}, gains R/B={meta['gains'][0]:.3f}/{meta['gains'][2]:.3f}"
          f" CCT={meta['cct']:.0f}K LS={meta['lsc_ls']}")
    print(f"-> {os.path.join(TDATA, 'golden_render.bin')} ({rgb8.size} bytes)")
    print(f"-> {os.path.join(TDATA, 'golden.json')}")
    print(f"-> {os.path.join(TDATA, 'golden.png')}")


if __name__ == "__main__":
    main()
