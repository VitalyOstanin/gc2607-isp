# Running gc2607-video as an on-demand systemd service

This directory holds the files to run `gc2607-video` as a system service that
exposes the GC2607 as a virtual webcam on demand: the loopback node stays present
so applications always see the camera, but the sensor and the ISP run only while
an application is actually capturing (driven by the v4l2loopback client-usage
event). While idle the camera is powered down, which also helps on a thermally
constrained laptop.

## Contents

- [Files](#files)
- [How it works](#how-it-works)
- [Install](#install)
- [Verify](#verify)
- [Reduce logging further](#reduce-logging-further)
- [Hardened variant (no root)](#hardened-variant-no-root)
- [Uninstall](#uninstall)

## Files

| File                                   | Installed to                                  | Purpose |
|----------------------------------------|-----------------------------------------------|---------|
| `systemd/gc2607-camera.service`        | `/etc/systemd/system/gc2607-camera.service`   | the on-demand daemon unit |
| `modules-load.d/gc2607-loopback.conf`  | `/etc/modules-load.d/gc2607-loopback.conf`    | load `v4l2loopback` at boot |
| `modprobe.d/gc2607-loopback.conf`      | `/etc/modprobe.d/gc2607-loopback.conf`        | loopback options (`card_label`, `exclusive_caps`) |

## How it works

- `v4l2loopback` is loaded at boot with `card_label="MateBook Camera (GC2607)"`
  and `exclusive_caps=1`. No `video_nr` is pinned: the IPU6 ISYS registers dozens
  of `/dev/video*` nodes, so a fixed number is fragile.
- The daemon finds the loopback by that label (`--device-label`), not by a
  `/dev/videoN` path, so it works regardless of probe order.
- `--on-demand auto` keeps the loopback open and starts the GC2607 only while a
  consumer captures; on disconnect the camera is stopped again.
- Telemetry is written to stdout and discarded by the unit (`StandardOutput=null`);
  warnings and errors go to stderr and land in the journal.

## Install

Build `gc2607-video` (see the top-level README — it is built in a podman
container) and install it plus the service files:

```sh
# 1. the binary on a stable path the unit refers to
sudo install -m 0755 gc2607-video /usr/local/bin/gc2607-video

# 2. loopback setup
sudo install -m 0644 packaging/modules-load.d/gc2607-loopback.conf /etc/modules-load.d/gc2607-loopback.conf
sudo install -m 0644 packaging/modprobe.d/gc2607-loopback.conf    /etc/modprobe.d/gc2607-loopback.conf
sudo modprobe v4l2loopback   # bring the node up now without rebooting

# 3. the service
sudo install -m 0644 packaging/systemd/gc2607-camera.service /etc/systemd/system/gc2607-camera.service
sudo systemctl daemon-reload
sudo systemctl enable --now gc2607-camera
```

The GC2607 V4L2 driver and the patched ipu-bridge must already be installed (see
the `gc2607-driver` and `gc2607-ipu-bridge` projects), and the physical camera
switch on the side of the laptop must be on.

## Verify

```sh
systemctl status gc2607-camera
v4l2-ctl --list-devices            # shows "MateBook Camera (GC2607)"
ffplay -f v4l2 /dev/videoN         # N from the list above; starts the camera on open
```

When nothing is capturing, `gc2607-camera` is running but the sensor is idle; the
camera spins up when an application opens the node and stops when it closes.

## Reduce logging further

`StandardOutput=null` already drops the periodic telemetry. If you also run the
binary by hand and want it quiet there, the telemetry is on stdout and errors on
stderr, so redirect stdout only:

```sh
gc2607-video --device-label "MateBook Camera (GC2607)" --on-demand auto >/dev/null
```

## Hardened variant (no root)

Instead of `User=root`, run as a transient user that is only in the groups needed
for device access. Replace the `User=root` line in the unit with:

```ini
DynamicUser=yes
SupplementaryGroups=video render
```

`video` covers the loopback, the IPU6 ISYS nodes and the sensor sub-device
(char-major 81); `render` covers `/dev/dri/renderD128` for the GPU backend. Verify
the sensor sub-device (`/dev/v4l-subdev*`) is group `video` on your system; if
not, an udev rule is needed, or keep `User=root`.

## Uninstall

```sh
sudo systemctl disable --now gc2607-camera
sudo rm /etc/systemd/system/gc2607-camera.service
sudo rm /etc/modules-load.d/gc2607-loopback.conf
sudo rm /etc/modprobe.d/gc2607-loopback.conf
sudo systemctl daemon-reload
sudo rmmod v4l2loopback        # optional: drop the loopback now
sudo rm /usr/local/bin/gc2607-video
```
