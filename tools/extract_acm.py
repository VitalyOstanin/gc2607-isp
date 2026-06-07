#!/usr/bin/env python3
"""Extract the advanced (sector) colour matrices from the camera's .aiqb
(CMC record id=25, cmc_name_id_advanced_color_matrices).

Layout (derived from ia_cmc_types.h and confirmed by the record size):
  ia_mkn_record_header (8): size u32, fmt u8, key u8, name u16 (== 25)
  num_light_srcs u16, num_sectors u16
  hue_of_sectors: u32[num_sectors]   (starting hue angle of each sector, deg)
  per light source (num_light_srcs):
    info (v101, 24 bytes): src_type u32, chromaticity[2] f32 (r/g, b/g),
                           cie_xy[2] f32, cct u32
    traditional: f32[9]                 (3x3, rows sum to 1)
    advanced:    f32[9*num_sectors]     (num_sectors 3x3 matrices, rows sum to 1)

Output: ../data/gc2607_acm.npz with keys
  hues          (S,) int32        sector start hue angles (deg)
  src_chroma    (L,2) float32     per-light-source [r/g, b/g]
  cct           (L,) int32        per-light-source colour temperature
  traditional   (L,3,3) float32   all-sector matrix per light source
  advanced      (L,S,3,3) float32 per-sector matrices per light source
"""
import os
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")

NAME_ID_ACM = 25


def main():
    if len(sys.argv) < 2:
        sys.exit(f"usage: {sys.argv[0]} <path-to.aiqb> [record_offset]")
    d = open(sys.argv[1], "rb").read()
    off = int(sys.argv[2]) if len(sys.argv) > 2 else 132024

    size, fmt, _key, name = struct.unpack_from("<IBBH", d, off)
    if name != NAME_ID_ACM:
        sys.exit(f"record at {off} has name id {name}, expected {NAME_ID_ACM}")

    nls, nsec = struct.unpack_from("<HH", d, off + 8)
    p = off + 12
    hues = np.array(struct.unpack_from(f"<{nsec}I", d, p), dtype=np.int32)
    p += nsec * 4

    src_chroma = np.zeros((nls, 2), dtype=np.float32)
    cct = np.zeros(nls, dtype=np.int32)
    traditional = np.zeros((nls, 3, 3), dtype=np.float32)
    advanced = np.zeros((nls, nsec, 3, 3), dtype=np.float32)

    for ls in range(nls):
        _src_type, rg, bg, _cx, _cy, ct = struct.unpack_from("<IffffI", d, p)
        p += 24
        src_chroma[ls] = (rg, bg)
        cct[ls] = ct
        traditional[ls] = np.array(struct.unpack_from("<9f", d, p), dtype=np.float32).reshape(3, 3)
        p += 36
        for s in range(nsec):
            advanced[ls, s] = np.array(struct.unpack_from("<9f", d, p), dtype=np.float32).reshape(3, 3)
            p += 36

    # Sanity: rows of every matrix sum to ~1 (luminance preserving).
    rs_t = traditional.sum(axis=2)
    rs_a = advanced.sum(axis=3)
    assert np.allclose(rs_t, 1.0, atol=1e-3), f"traditional rows not ~1: {rs_t.min()}..{rs_t.max()}"
    assert np.allclose(rs_a, 1.0, atol=1e-3), f"advanced rows not ~1: {rs_a.min()}..{rs_a.max()}"
    assert np.all(np.diff(hues) > 0), f"hues not increasing: {hues}"
    consumed = p - (off + 8)
    assert consumed <= size, f"consumed {consumed} > record size {size}"

    os.makedirs(DATA, exist_ok=True)
    out = os.path.join(DATA, "gc2607_acm.npz")
    np.savez(out, hues=hues, src_chroma=src_chroma, cct=cct,
             traditional=traditional, advanced=advanced)

    print(f"ACM record @ {off}: size={size} fmt={fmt} name={name}")
    print(f"light sources={nls} sectors={nsec}")
    print(f"hues(deg)={hues.tolist()}")
    for ls in range(nls):
        print(f"  LS{ls}: chroma=({src_chroma[ls,0]:.3f},{src_chroma[ls,1]:.3f}) "
              f"cct={cct[ls]}K traditional=\n{traditional[ls]}")
    print(f"advanced matrices: {advanced.shape}, "
          f"value range {advanced.min():.3f}..{advanced.max():.3f}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
