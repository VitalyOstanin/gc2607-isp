#!/usr/bin/env python3
"""Extract the lateral chromatic aberration correction grids from the camera's
.aiqb (CMC record id=29, cmc_name_id_lateral_chromatic_aberration_correction).

The .aiqb is the camera's tuning container; its path is passed as an argument
(it is not part of this repository). The CMC record layout follows
ia_cmc_types.h / ia_mkn_types.h:

  ia_mkn_record_header (8 bytes): size u32, data_format_id u8, key_id u8,
                                  data_name_id u16 (== 29)
  optical_center: cmc_coords = x u16, y u16    (no-aberration location)
  grid_width u16, grid_height u16
  cell_size_x u16, cell_size_y u16             (px between grid points, 2^n)
  then 4 float grids, each grid_width*grid_height, row-major:
    lca_grid_blue_x, lca_grid_blue_y, lca_grid_red_x, lca_grid_red_y
  Each value is the absolute lateral shift (in native-resolution pixels) of the
  blue/red pixel relative to the green pixel at that grid location (+x right,
  +y down). The grid is evenly spaced over the native maximum sensor resolution.

Output: ../data/gc2607_lca.npz with keys
  grids       (4, gh, gw) float32, order [blue_x, blue_y, red_x, red_y]
  optical_center (2,) int32  [x, y]
  cell_size      (2,) int32  [x, y]
"""
import os
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")

NAME_ID_LCA = 29


def main():
    if len(sys.argv) < 2:
        sys.exit(f"usage: {sys.argv[0]} <path-to.aiqb> [record_offset]")
    aiqb = sys.argv[1]
    d = open(aiqb, "rb").read()

    # Locate the LCA record. Default to the known offset, but verify the header;
    # an explicit offset can be passed as argv[2].
    off = int(sys.argv[2]) if len(sys.argv) > 2 else 136816
    size, fmt, _key, name = struct.unpack_from("<IBBH", d, off)
    if name != NAME_ID_LCA:
        # Fall back to scanning for the record by name id.
        found = None
        for o in range(0, len(d) - 8):
            s, _f, _k, n = struct.unpack_from("<IBBH", d, o)
            if n == NAME_ID_LCA and 20 < s < len(d) and o + s <= len(d):
                # plausible header + 4 float grids
                ox, oy, gw, gh, cx, cy = struct.unpack_from("<HHHHHH", d, o + 8)
                if gw * gh * 16 + 20 == s and 0 < gw < 256 and 0 < gh < 256:
                    found = o
                    break
        if found is None:
            sys.exit(f"LCA record (name id {NAME_ID_LCA}) not found")
        off = found
        size, fmt, _key, name = struct.unpack_from("<IBBH", d, off)

    ox, oy, gw, gh, cx, cy = struct.unpack_from("<HHHHHH", d, off + 8)
    n = gw * gh
    base = off + 20
    keys = ["blue_x", "blue_y", "red_x", "red_y"]
    grids = np.stack([
        np.frombuffer(d, "<f4", count=n, offset=base + i * n * 4).reshape(gh, gw)
        for i in range(4)
    ]).astype("<f4")

    # The record size includes the 20-byte header+geometry, the four float grids,
    # and up to 7 bytes of trailing padding to 8-byte alignment.
    data_size = 20 + 4 * n * 4
    assert data_size <= size < data_size + 8, f"size {size} vs data {data_size}"

    os.makedirs(DATA, exist_ok=True)
    out = os.path.join(DATA, "gc2607_lca.npz")
    np.savez(
        out,
        grids=grids,
        optical_center=np.array([ox, oy], dtype=np.int32),
        cell_size=np.array([cx, cy], dtype=np.int32),
    )

    print(f"LCA record @ {off}: size={size} fmt={fmt} name={name}")
    print(f"optical_center=({ox},{oy}) grid={gw}x{gh} cell=({cx},{cy})")
    for i, k in enumerate(keys):
        a = grids[i]
        print(f"  {k}: min={a.min():.4f} max={a.max():.4f} mean={a.mean():.4f} (px)")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
