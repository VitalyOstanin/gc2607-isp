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

/// Compute the next sensor state from the current state and the brightness
/// `metric` (mean linear signal as a fraction of full scale, 0..1) measured on
/// the frame captured with `state`.
///
/// Returns `state` unchanged when the metric is within the deadband.
pub fn step(cfg: &AeConfig, state: AeState, metric: f64) -> AeState {
    let metric = metric.max(1e-4);
    let ratio = cfg.target / metric;

    // Converged: leave the sensor untouched to avoid wobble.
    if (1.0 - cfg.deadband..=1.0 + cfg.deadband).contains(&ratio) {
        return state;
    }

    // Damp and clamp the per-frame correction.
    let r = ratio
        .powf(cfg.damping)
        .clamp(1.0 / cfg.max_step, cfg.max_step);

    let cur_mult = GAIN_TABLE[state.gain_index as usize];
    let cur_product = state.exposure as f64 * cur_mult;
    // Desired exposure-time * gain product, in (lines * multiplier) units.
    let target_product = (cur_product * r).max(EXPOSURE_MIN as f64);

    let vblank_floor = vblank_for_fps(cfg.min_fps);
    let exp_cap_30fps = exposure_max(VBLANK_MIN) as f64;
    let exp_cap_floor = exposure_max(vblank_floor);

    let (exposure, vblank, gain_index) = if target_product <= exp_cap_30fps {
        // Fits within 30 fps at unity gain.
        let e = (target_product.round() as i32).clamp(EXPOSURE_MIN, exposure_max(VBLANK_MIN));
        (e, VBLANK_MIN, 0u8)
    } else if target_product <= exp_cap_floor as f64 {
        // Extend the frame (lower fps) at unity gain before touching gain.
        let needed_vts = target_product.round() as i32 + EXPOSURE_MARGIN;
        let vb = (needed_vts - NATIVE_HEIGHT).clamp(VBLANK_MIN, vblank_floor);
        let e = (target_product.round() as i32).clamp(EXPOSURE_MIN, exposure_max(vb));
        (e, vb, 0u8)
    } else {
        // Exposure saturated at the fps floor: make up the rest with gain.
        let e = exp_cap_floor;
        let remaining = target_product / e as f64;
        let g = pick_gain_index(remaining).min(cfg.max_gain_index);
        (e, vblank_floor, g)
    };

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

#[cfg(test)]
mod tests {
    use super::*;

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
