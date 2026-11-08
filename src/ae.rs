//! Auto-exposure metering for the GC2607.
//!
//! libcamera's soft IPA exposes no manual exposure control and cannot disable
//! its own AGC, so exposure and gain are metered here from our own ISP
//! statistics and written directly to the sensor subdev (see [`crate::sensor`]).
//!
//! The policy is exposure-priority: to add light, lengthen the integration time
//! first — extending the frame through vertical blanking down to a configurable
//! minimum frame rate — and only then raise analogue gain, which adds noise.
//! To remove light the same allocation runs in reverse, draining gain before
//! exposure. The control loop is closed (metric measured from the previous
//! frame), so a damped step converges in a few frames without oscillation.
//!
//! This module is pure arithmetic with no I/O, so the allocation policy is unit
//! tested directly.

/// Analogue gain multipliers per LUT index (0..=16).
///
/// The sensor exposes `V4L2_CID_ANALOGUE_GAIN` as a look-up table index, not a
/// register value. The multipliers are the combined digital gain
/// `(dgain_int << 8 | dgain_frac) / 64` from the kernel driver's gain table
/// (the same table backs the libcamera `CameraSensorHelper`). It is a geometric
/// progression that doubles every four steps (1.0, 2.0, 4.0, 8.0, 16.0); this
/// was confirmed empirically (index 8 yields ~4x the mean signal).
// The first entry is written `64.0 / 64.0` (not `1.0`) to keep the raw register
// numerator visible in the column, like every other row; clippy::eq_op flags the
// identical operands, which is intentional here.
#[allow(clippy::eq_op)]
pub const GAIN_TABLE: [f64; 17] = [
    64.0 / 64.0,
    75.0 / 64.0,
    89.0 / 64.0,
    106.0 / 64.0,
    128.0 / 64.0,
    151.0 / 64.0,
    179.0 / 64.0,
    212.0 / 64.0,
    256.0 / 64.0,
    303.0 / 64.0,
    358.0 / 64.0,
    424.0 / 64.0,
    512.0 / 64.0,
    606.0 / 64.0,
    716.0 / 64.0,
    848.0 / 64.0,
    1024.0 / 64.0,
];

/// Highest valid analogue-gain LUT index (`GAIN_TABLE` has 17 entries, 0..=16).
pub const MAX_GAIN_INDEX: u8 = (GAIN_TABLE.len() - 1) as u8;

/// Active line count used for the frame-length (VTS) computation.
pub const NATIVE_HEIGHT: i32 = 1088;
/// Horizontal total size (line length in pixel-clock units): width + hblank.
pub const HTS: i32 = 2745;
/// Sensor pixel rate (Hz), from the subdev's read-only `PIXEL_RATE` control.
pub const PIXEL_RATE: f64 = 102_937_500.0;
/// Default frame length (lines): 30 fps.
pub const VTS_DEFAULT: i32 = 1250;
/// Shortest vertical blanking (30 fps): `VTS_DEFAULT - NATIVE_HEIGHT`.
pub const VBLANK_MIN: i32 = VTS_DEFAULT - NATIVE_HEIGHT;
/// Largest frame length the sensor accepts (14-bit register).
pub const VTS_MAX: i32 = 0x3fff;
/// Largest vertical blanking: `VTS_MAX - NATIVE_HEIGHT`.
pub const VBLANK_MAX: i32 = VTS_MAX - NATIVE_HEIGHT;
/// Minimum integration time (lines).
pub const EXPOSURE_MIN: i32 = 4;
/// Exposure must stay this many lines below the frame length.
pub const EXPOSURE_MARGIN: i32 = 4;

/// Frame rate (fps) for a given vertical blanking.
pub fn frame_rate(vblank: i32) -> f64 {
    PIXEL_RATE / (HTS as f64 * (NATIVE_HEIGHT + vblank) as f64)
}

/// Maximum integration time (lines) at a given vertical blanking. Mirrors the
/// kernel driver: `exposure_max = NATIVE_HEIGHT + vblank - EXPOSURE_MARGIN`.
pub fn exposure_max(vblank: i32) -> i32 {
    NATIVE_HEIGHT + vblank - EXPOSURE_MARGIN
}

/// Smallest vertical blanking whose frame rate is still at or above `fps`,
/// clamped to the sensor's range.
pub fn vblank_for_fps(fps: f64) -> i32 {
    let vts = (PIXEL_RATE / (HTS as f64 * fps)).floor() as i32;
    (vts - NATIVE_HEIGHT).clamp(VBLANK_MIN, VBLANK_MAX)
}

/// Index of the gain table entry whose multiplier is closest to `mult`.
pub fn pick_gain_index(mult: f64) -> u8 {
    GAIN_TABLE
        .iter()
        .enumerate()
        .min_by(|(_, &a), (_, &b)| (a - mult).abs().total_cmp(&(b - mult).abs()))
        .map(|(i, _)| i as u8)
        .unwrap_or(0)
}

/// Sensor exposure state the AE loop reads and writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AeState {
    /// Integration time in lines.
    pub exposure: i32,
    /// Analogue gain LUT index (0..=16).
    pub gain_index: u8,
    /// Vertical blanking in lines (sets the frame length, hence the frame rate).
    pub vblank: i32,
}

impl Default for AeState {
    fn default() -> Self {
        AeState {
            exposure: EXPOSURE_MIN.max(VTS_DEFAULT / 2),
            gain_index: 0,
            vblank: VBLANK_MIN,
        }
    }
}

/// AE loop tuning.
#[derive(Debug, Clone, Copy)]
pub struct AeConfig {
    /// Target metric, as a fraction of full scale (0..1).
    pub target: f64,
    /// Relative deadband half-width: skip adjustment while the correction ratio
    /// stays within `[1 - deadband, 1 + deadband]` (prevents wobble).
    pub deadband: f64,
    /// Exponent applied to the correction ratio for damping (0..1; lower is
    /// slower but steadier).
    pub damping: f64,
    /// Largest single-frame change factor for the exposure-gain product.
    pub max_step: f64,
    /// Frame rate floor: exposure may lengthen (lowering fps) only down to this
    /// rate before gain is used instead.
    pub min_fps: f64,
    /// Highest gain index the loop may select (noise ceiling).
    pub max_gain_index: u8,
}

impl Default for AeConfig {
    fn default() -> Self {
        AeConfig {
            target: 0.35,
            deadband: 0.12,
            damping: 0.7,
            max_step: 8.0,
            min_fps: 15.0,
            max_gain_index: 16,
        }
    }
}

/// Split an exposure-time x gain product (lines x multiplier) into the sensor
/// state that delivers it: exposure first within 30 fps, then a longer frame
/// down to the `min_fps` floor, then gain (capped at `max_gain_index`).
fn split_product(cfg: &AeConfig, product: f64) -> AeState {
    let product = product.max(EXPOSURE_MIN as f64);
    let vblank_floor = vblank_for_fps(cfg.min_fps);
    let exp_cap_30fps = exposure_max(VBLANK_MIN) as f64;
    let exp_cap_floor = exposure_max(vblank_floor);

    let (exposure, vblank, gain_index) = if product <= exp_cap_30fps {
        // Fits within 30 fps at unity gain.
        let e = (product.round() as i32).clamp(EXPOSURE_MIN, exposure_max(VBLANK_MIN));
        (e, VBLANK_MIN, 0u8)
    } else if product <= exp_cap_floor as f64 {
        // Extend the frame (lower fps) at unity gain before touching gain.
        let needed_vts = product.round() as i32 + EXPOSURE_MARGIN;
        let vb = (needed_vts - NATIVE_HEIGHT).clamp(VBLANK_MIN, vblank_floor);
        let e = (product.round() as i32).clamp(EXPOSURE_MIN, exposure_max(vb));
        (e, vb, 0u8)
    } else {
        // Exposure saturated at the fps floor: make up the rest with gain.
        let e = exp_cap_floor;
        let g = pick_gain_index(product / e as f64).min(cfg.max_gain_index);
        (e, vblank_floor, g)
    };
    AeState {
        exposure,
        gain_index,
        vblank,
    }
}

/// Compute the next sensor state from the current state and the brightness
/// `metric` (mean linear signal as a fraction of full scale, 0..1) measured on
/// the frame captured with `state`.
///
/// Within the deadband the brightness is left alone, but a state whose frame
/// length or gain does not match the current `min_fps` floor (one inherited
/// from the sensor, or set under another floor) is re-split at the same
/// product; otherwise `state` is returned unchanged.
pub fn step(cfg: &AeConfig, state: AeState, metric: f64) -> AeState {
    let metric = metric.max(1e-4);
    let ratio = cfg.target / metric;

    let cur_product = state.exposure as f64 * GAIN_TABLE[state.gain_index as usize];

    // Converged: leave the brightness untouched to avoid wobble.
    if (1.0 - cfg.deadband..=1.0 + cfg.deadband).contains(&ratio) {
        let even = split_product(cfg, cur_product);
        if even.vblank != state.vblank || even.gain_index != state.gain_index {
            return even;
        }
        return state;
    }

    // Damp and clamp the per-frame correction.
    let r = ratio
        .powf(cfg.damping)
        .clamp(1.0 / cfg.max_step, cfg.max_step);

    // Desired exposure-time * gain product, in (lines * multiplier) units.
    let AeState {
        exposure,
        gain_index,
        vblank,
    } = split_product(cfg, cur_product * r);

    let mut next = AeState {
        exposure,
        gain_index,
        vblank,
    };

    // Outside the deadband but integer rounding of the exposure produced no
    // change: nudge one line (or one gain step at the exposure cap) toward the
    // target so the loop cannot stall just shy of the deadband. Each nudge is a
    // sub-percent brightness change, so it converges into the deadband without
    // overshoot. Reachable only for tiny exposures; cheap insurance regardless.
    if next == state {
        if ratio > 1.0 {
            if next.exposure < exposure_max(next.vblank) {
                next.exposure += 1;
            } else if next.gain_index < cfg.max_gain_index {
                next.gain_index += 1;
            }
        } else if next.exposure > EXPOSURE_MIN {
            next.exposure -= 1;
        } else if next.gain_index > 0 {
            next.gain_index -= 1;
        }
    }

    next
}

/// Output luma code at or above which a sample counts as clipped (full-range
/// 8-bit output, so 255 is white; a few codes of margin catch highlights the
/// tone curve has already flattened).
pub const CLIP_LEVEL: u8 = 250;

/// Highlight protection settings.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HighlightConfig {
    /// Largest tolerated fraction of clipped samples (0 disables the guard).
    pub clip_max: f64,
    /// Lowest factor the guard may apply to the AE target.
    pub floor: f64,
}

/// Default largest tolerated clipped fraction.
pub const CLIP_MAX: f64 = 0.02;
/// Default lowest target factor of the highlight guard: 0.36 goes no lower
/// than about 0.22, where a face still reads well in daylight.
pub const CLIP_FLOOR: f64 = 0.6;

impl Default for HighlightConfig {
    fn default() -> Self {
        HighlightConfig {
            clip_max: CLIP_MAX,
            floor: CLIP_FLOOR,
        }
    }
}

/// Per-decision factor by which the guard lowers the target while too much
/// of the frame is clipped.
const GUARD_DOWN: f64 = 0.92;
/// Per-decision factor by which the guard gives the target back once the
/// highlights are well clear; slower than the descent.
const GUARD_UP: f64 = 1.02;
/// The guard releases only below this share of `clip_max`; between it and
/// `clip_max` the factor holds (hysteresis).
const GUARD_RELEASE: f64 = 0.25;
/// Per-decision growth of the release ceiling: about +8 % a minute at 25
/// decisions a second. In a scene that keeps clipping, the release tries one
/// step brighter about once a minute; one that stopped clipping (a white
/// shirt left the frame) regains the full target within several minutes.
const CEILING_RELAX: f64 = 1.00005;
/// A scene this much darker (same exposure, lower metric) than the one that
/// clipped lifts the release ceiling at once.
const SCENE_DARKER: f64 = 0.7;

/// Highlight protection: lowers the AE target while more than `clip_max` of
/// the frame is clipped, down to `floor` times the target, and gives it back
/// slowly once the clipped share falls well below the limit. A face-weighted
/// target such as 0.36 suits a dim room but burns a white shirt or a window
/// in daylight; this keeps the brighter target where it is harmless.
///
/// The AE deadband moves the exposure in steps of at least 12 %, and in a
/// scene with a large bright area one such step can carry the clipped share
/// from above the limit to below the release level. Giving the target back
/// would then clip again and pump. The guard therefore remembers the factor
/// at which the frame last clipped and releases only to just below it; that
/// ceiling lifts slowly with time, and at once when the scene gets darker.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HighlightGuard {
    scale: f64,
    /// Factor at which the frame last clipped; the release stops one
    /// `GUARD_DOWN` step below it. `1 / GUARD_DOWN` means no limit.
    ceiling: f64,
    /// Scene brightness (metric per unit exposure-gain product) when the
    /// frame last clipped.
    clip_scene: f64,
}

impl Default for HighlightGuard {
    fn default() -> Self {
        HighlightGuard {
            scale: 1.0,
            ceiling: 1.0 / GUARD_DOWN,
            clip_scene: 0.0,
        }
    }
}

impl HighlightGuard {
    /// The factor currently applied to the AE target (`floor..=1`).
    pub fn scale(&self) -> f64 {
        self.scale
    }

    /// Update the factor from the frame metered at `state`: its clipped
    /// fraction and its brightness `metric`. Returns the new factor.
    pub fn update(
        &mut self,
        cfg: &HighlightConfig,
        state: AeState,
        metric: f64,
        clipped: f64,
    ) -> f64 {
        let floor = cfg.floor.clamp(0.0, 1.0);
        if cfg.clip_max <= 0.0 {
            *self = HighlightGuard::default();
            return 1.0;
        }
        let product = state.exposure as f64 * GAIN_TABLE[state.gain_index as usize];
        let scene = metric / product.max(1.0);
        if clipped > cfg.clip_max {
            self.ceiling = self.scale.min(self.ceiling);
            self.clip_scene = scene;
            self.scale = (self.scale * GUARD_DOWN).max(floor);
        } else if clipped < cfg.clip_max * GUARD_RELEASE {
            let open = 1.0 / GUARD_DOWN;
            self.ceiling = if scene < self.clip_scene * SCENE_DARKER {
                open
            } else {
                (self.ceiling * CEILING_RELAX).min(open)
            };
            let cap = (self.ceiling * GUARD_DOWN).min(1.0);
            if self.scale < cap {
                self.scale = (self.scale * GUARD_UP).min(cap);
            }
        }
        self.scale = self.scale.max(floor);
        self.scale
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ST: AeState = AeState {
        exposure: 600,
        gain_index: 0,
        vblank: VBLANK_MIN,
    };

    #[test]
    fn guard_lowers_target_down_to_the_floor() {
        let cfg = HighlightConfig::default();
        let mut g = HighlightGuard::default();
        for _ in 0..100 {
            g.update(&cfg, ST, 0.36, 0.12);
        }
        assert!((g.scale() - cfg.floor).abs() < 1e-12, "{g:?}");
    }

    #[test]
    fn guard_holds_inside_the_hysteresis_band() {
        let cfg = HighlightConfig::default();
        let mut g = HighlightGuard::default();
        g.update(&cfg, ST, 0.36, 0.05);
        g.update(&cfg, ST, 0.36, 0.05);
        let s = g.scale();
        assert!(s < 1.0);
        // Between clip_max / 4 and clip_max the factor holds.
        for clipped in [0.006, 0.01, 0.02] {
            assert_eq!(g.update(&cfg, ST, 0.36, clipped), s, "{clipped}");
        }
    }

    #[test]
    fn guard_releases_below_the_factor_that_clipped() {
        let cfg = HighlightConfig::default();
        let mut g = HighlightGuard::default();
        // Clips at full target, then at 0.92; settles one step lower.
        g.update(&cfg, ST, 0.36, 0.05);
        g.update(&cfg, ST, 0.33, 0.05);
        let clipped_at = 0.92;
        for _ in 0..50 {
            g.update(&cfg, ST, 0.30, 0.0);
        }
        let cap = clipped_at * GUARD_DOWN;
        assert!(g.scale() <= cap * CEILING_RELAX.powi(50) + 1e-12, "{g:?}");
        assert!(g.scale() > cap * 0.99, "{g:?}");
    }

    #[test]
    fn guard_ceiling_lifts_with_time() {
        let cfg = HighlightConfig::default();
        let mut g = HighlightGuard::default();
        for _ in 0..100 {
            g.update(&cfg, ST, 0.36, 0.2);
        }
        let mut steps = 0;
        while g.scale() < 1.0 {
            // Same scene (same metric at the same exposure), clear of clipping.
            g.update(&cfg, ST, 0.36, 0.0);
            steps += 1;
            assert!(steps < 50_000, "{g:?}");
        }
        // Minutes at 25 decisions a second, not a few frames.
        assert!(steps > 1_000, "{steps}");
    }

    #[test]
    fn guard_releases_at_once_when_the_scene_darkens() {
        let cfg = HighlightConfig::default();
        let mut g = HighlightGuard::default();
        for _ in 0..100 {
            g.update(&cfg, ST, 0.36, 0.2);
        }
        let mut steps = 0;
        while g.scale() < 1.0 {
            // Same exposure, half the brightness: the light went down.
            g.update(&cfg, ST, 0.18, 0.0);
            steps += 1;
        }
        assert!(steps < 40, "{steps}");
    }

    #[test]
    fn guard_off_keeps_the_target() {
        let cfg = HighlightConfig {
            clip_max: 0.0,
            ..HighlightConfig::default()
        };
        let mut g = HighlightGuard::default();
        assert_eq!(g.update(&cfg, ST, 0.36, 0.9), 1.0);
    }

    /// Run the guarded loop in a scene where a bright area clips as the
    /// exposure grows; `clipped_at` maps the metric to the clipped share.
    fn run_guarded(clipped_at: impl Fn(f64) -> f64) -> Vec<(AeState, f64, f64)> {
        let ae_cfg = AeConfig {
            target: 0.36,
            ..AeConfig::default()
        };
        let hl = HighlightConfig::default();
        let k = 0.36 / 900.0; // metric = k * product
        let mut g = HighlightGuard::default();
        let mut st = AeState::default();
        let mut history = Vec::new();
        for _ in 0..1000 {
            let product = st.exposure as f64 * GAIN_TABLE[st.gain_index as usize];
            let m = (k * product).min(1.0);
            let scale = g.update(&hl, st, m, clipped_at(m));
            let cfg = AeConfig {
                target: ae_cfg.target * scale,
                ..ae_cfg
            };
            st = step(&cfg, st, m);
            history.push((st, m, clipped_at(m)));
        }
        history
    }

    #[test]
    fn guarded_loop_settles_in_a_steeply_clipping_scene() {
        // One AE step carries the clipped share across the whole hysteresis
        // band: the ceiling must stop the loop from pumping.
        let h = run_guarded(|m| (0.25 * (m / 0.36 - 0.7) / 0.3).clamp(0.0, 0.25));
        let tail = &h[h.len() - 300..];
        assert!(tail.iter().all(|s| s.0 == tail[0].0), "{:?}", &tail[..5]);
        assert!(tail[0].1 < 0.36 * 0.9, "metric {}", tail[0].1);
        assert!(tail[0].2 <= 0.02, "clipped {}", tail[0].2);
    }

    #[test]
    fn guarded_loop_keeps_the_target_without_highlights() {
        let h = run_guarded(|_| 0.0);
        let m = h.last().unwrap().1;
        assert!((m / 0.36 - 1.0).abs() <= 0.12, "metric {m}");
    }

    #[test]
    fn gain_table_anchors() {
        assert!((GAIN_TABLE[0] - 1.0).abs() < 1e-9);
        assert!((GAIN_TABLE[4] - 2.0).abs() < 1e-9);
        assert!((GAIN_TABLE[8] - 4.0).abs() < 1e-9);
        assert!((GAIN_TABLE[12] - 8.0).abs() < 1e-9);
        assert!((GAIN_TABLE[16] - 16.0).abs() < 1e-9);
    }

    #[test]
    fn frame_rate_default_is_30fps() {
        assert!((frame_rate(VBLANK_MIN) - 30.0).abs() < 0.05);
    }

    #[test]
    fn pick_gain_nearest() {
        assert_eq!(pick_gain_index(1.0), 0);
        assert_eq!(pick_gain_index(4.0), 8);
        assert_eq!(pick_gain_index(100.0), 16); // clamps to the top entry
        assert_eq!(pick_gain_index(0.1), 0); // clamps to unity
    }

    #[test]
    fn converged_metric_holds_state() {
        let cfg = AeConfig::default();
        let st = AeState {
            exposure: 600,
            gain_index: 0,
            vblank: VBLANK_MIN,
        };
        // metric == target -> ratio 1.0, inside deadband.
        assert_eq!(step(&cfg, st, cfg.target), st);
    }

    #[test]
    fn converged_inherited_30fps_split_moves_to_the_fps_floor() {
        // The sensor kept a 30 fps exposure with gain from a session run at a
        // higher fps floor; on target brightness the split must still follow
        // the current floor (longer exposure, less gain).
        let cfg = AeConfig {
            min_fps: 24.0,
            ..AeConfig::default()
        };
        let st = AeState {
            exposure: exposure_max(VBLANK_MIN),
            gain_index: 11,
            vblank: VBLANK_MIN,
        };
        let next = step(&cfg, st, cfg.target);
        assert_eq!(next.vblank, vblank_for_fps(24.0));
        assert_eq!(next.exposure, exposure_max(vblank_for_fps(24.0)));
        assert!(next.gain_index < st.gain_index, "{next:?}");
    }

    #[test]
    fn converged_frame_longer_than_the_floor_shortens() {
        // The floor was raised (e.g. 15 -> 30 fps): a converged state running
        // slower than the floor must come back to it.
        let cfg = AeConfig {
            min_fps: 30.0,
            ..AeConfig::default()
        };
        let st = AeState {
            exposure: exposure_max(vblank_for_fps(15.0)),
            gain_index: 2,
            vblank: vblank_for_fps(15.0),
        };
        let next = step(&cfg, st, cfg.target);
        assert_eq!(next.vblank, VBLANK_MIN);
        assert!(next.gain_index > st.gain_index, "{next:?}");
    }

    #[test]
    fn converged_split_at_the_floor_holds() {
        let cfg = AeConfig {
            min_fps: 24.0,
            ..AeConfig::default()
        };
        let vb = vblank_for_fps(24.0);
        let st = AeState {
            exposure: exposure_max(vb),
            gain_index: 10,
            vblank: vb,
        };
        assert_eq!(step(&cfg, st, cfg.target), st);
        // A converged state that needs no gain holds too, whatever its frame.
        let short = AeState {
            exposure: 900,
            gain_index: 0,
            vblank: VBLANK_MIN,
        };
        assert_eq!(step(&cfg, short, cfg.target), short);
    }

    #[test]
    fn dark_scene_settles_without_oscillation_at_a_lowered_floor() {
        // Start from a 30 fps state with gain (as inherited from the sensor)
        // in a scene that needs gain even at the 24 fps floor; once settled,
        // the state must stop changing.
        let cfg = AeConfig {
            min_fps: 24.0,
            target: 0.36,
            ..AeConfig::default()
        };
        let k = cfg.target / 12_000.0; // metric = k * product
        let mut st = AeState {
            exposure: exposure_max(VBLANK_MIN),
            gain_index: 11,
            vblank: VBLANK_MIN,
        };
        let mut history = Vec::new();
        for _ in 0..80 {
            let product = st.exposure as f64 * GAIN_TABLE[st.gain_index as usize];
            st = step(&cfg, st, (k * product).min(1.0));
            history.push(st);
        }
        let tail = &history[history.len() - 30..];
        assert!(tail.iter().all(|s| *s == tail[0]), "{tail:?}");
        assert_eq!(tail[0].vblank, vblank_for_fps(24.0));
    }

    #[test]
    fn too_dark_raises_exposure_before_gain() {
        let cfg = AeConfig::default();
        let st = AeState {
            exposure: 100,
            gain_index: 0,
            vblank: VBLANK_MIN,
        };
        // Half the target brightness -> needs more light, still fits at 30 fps.
        let next = step(&cfg, st, cfg.target / 2.0);
        assert!(next.exposure > st.exposure);
        assert_eq!(next.gain_index, 0);
        assert_eq!(next.vblank, VBLANK_MIN);
    }

    #[test]
    fn very_dark_extends_frame_then_uses_gain() {
        let cfg = AeConfig::default();
        // Start already near the 30 fps exposure cap.
        let st = AeState {
            exposure: exposure_max(VBLANK_MIN),
            gain_index: 0,
            vblank: VBLANK_MIN,
        };
        // Very dark: needs far more light than one 30 fps frame can give.
        let next = step(&cfg, st, 0.01);
        assert!(next.vblank > VBLANK_MIN, "frame should be extended");
        assert!(frame_rate(next.vblank) >= cfg.min_fps - 0.01);
    }

    #[test]
    fn too_bright_reduces_light() {
        let cfg = AeConfig::default();
        let st = AeState {
            exposure: 1200,
            gain_index: 8,
            vblank: VBLANK_MIN,
        };
        // Way over target -> product must shrink.
        let before = st.exposure as f64 * GAIN_TABLE[st.gain_index as usize];
        let next = step(&cfg, st, cfg.target * 4.0);
        let after = next.exposure as f64 * GAIN_TABLE[next.gain_index as usize];
        assert!(after < before);
    }

    #[test]
    fn converges_with_nonlinear_metric() {
        // The live loop now meters linear luminance (the output luma with the
        // sRGB gamma inverted), so the metric matches step()'s linear model. This
        // test keeps a non-linear response — metric = (k * product)^(1/2.4), the
        // pre-linearization regime — as a robustness guard: step() must still
        // converge on any monotonic metric, so a future metric change cannot
        // silently break the loop.
        let cfg = AeConfig::default();
        let mut st = AeState::default();
        let k = 3.0e-4f64; // scene gain; converged product well within 30 fps range
        for _ in 0..120 {
            let product = st.exposure as f64 * GAIN_TABLE[st.gain_index as usize];
            let metric = (k * product).powf(1.0 / 2.4).min(1.0);
            st = step(&cfg, st, metric);
        }
        let product = st.exposure as f64 * GAIN_TABLE[st.gain_index as usize];
        let metric = (k * product).powf(1.0 / 2.4);
        assert!(
            (metric - cfg.target).abs() < cfg.target * cfg.deadband * 1.5,
            "non-linear metric {metric} should converge near target {}",
            cfg.target
        );
    }

    #[test]
    fn converges_in_a_few_frames() {
        let cfg = AeConfig::default();
        let mut st = AeState::default();
        // Simulate a scene where true brightness == k * exposure_product.
        // Pick k so that the converged product is well within 30 fps range.
        let k = cfg.target / 4000.0; // metric = k * product
        for _ in 0..40 {
            let product = st.exposure as f64 * GAIN_TABLE[st.gain_index as usize];
            let metric = (k * product).min(1.0);
            st = step(&cfg, st, metric);
        }
        let product = st.exposure as f64 * GAIN_TABLE[st.gain_index as usize];
        let metric = k * product;
        assert!(
            (metric - cfg.target).abs() < cfg.target * cfg.deadband * 1.5,
            "metric {metric} should be near target {}",
            cfg.target
        );
    }
}
