#!/usr/bin/env python3
"""Extract the sensor's white response to reference light sources from the
camera's .aiqb (CMC record id=15, cmc_name_id_chromaticity_response).

Layout (ia_cmc_types.h, cmc_chromaticity_response_t):
  ia_mkn_record_header (8): size u32, fmt u8, key u8, name u16 (== 15)
  num_lightsources u16, num_nvm_lightsources u16
  lightsources in the order avg, high, low (num_lightsources each):
    cie x u16, cie y u16 (/65535), r/g u16, b/g u16 (/1024)

The r/g, b/g scale is not documented; /1024 matches the float chromaticity
the advanced colour matrix record (id=25) stores for the same CIE points.
Only the average set is kept (high/low are the module-to-module spread).

These six points run from about 2300 K to 7500 K along the Planckian locus,
wider than the five CCM calibration points (2963..6426 K, two of them
fluorescent). The AWB uses them as a second white-locus polyline.

Output: ../data/gc2607_awb_locus.json, a list of
  {"cie": [x, y], "chroma": [r/g, b/g]} sorted from warm to cool.
"""
import json
import os
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")

NAME_ID = 15


def find_record(d, name_id):
    # CMC records sit back to back after the LCMC group header.
    start = d.find(b"LCMC")
    if start < 0:
        sys.exit("no LCMC group")
    for o in range(start, start + 0x80, 4):
        size, fmt, _key, name = struct.unpack_from("<IBBH", d, o)
        if fmt in (100, 101, 102, 200) and 8 <= size < len(d):
            off = o
            break
    else:
        sys.exit("no CMC record after LCMC")
    while off + 8 <= len(d):
        size, _fmt, _key, name = struct.unpack_from("<IBBH", d, off)
        if size < 8:
            break
        if name == name_id:
            return off
        off += size
    sys.exit(f"record {name_id} not found")


def main():
    if len(sys.argv) < 2:
        sys.exit(f"usage: {sys.argv[0]} <path-to.aiqb>")
    d = open(sys.argv[1], "rb").read()
    off = find_record(d, NAME_ID)
    n, _nvm = struct.unpack_from("<HH", d, off + 8)
    pts = []
    for i in range(n):
        x, y, rg, bg = struct.unpack_from("<4H", d, off + 12 + 8 * i)
        pts.append({"cie": [x / 65535, y / 65535], "chroma": [rg / 1024, bg / 1024]})
    pts.sort(key=lambda p: -p["chroma"][0])
    out = os.path.join(DATA, "gc2607_awb_locus.json")
    with open(out, "w") as f:
        json.dump(pts, f, indent=1)
        f.write("\n")
    for p in pts:
        print(f"cie=({p['cie'][0]:.4f},{p['cie'][1]:.4f}) "
              f"rg={p['chroma'][0]:.4f} bg={p['chroma'][1]:.4f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
