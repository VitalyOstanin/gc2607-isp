//! AWB robustness: a large saturated colour object must not bias the white
//! balance away from the scene's neutral surfaces.
//!
//! Reproduces the reported defect (neutral whites acquire a blue cast when a
//! big saturated yellow object dominates the frame). The gray-world estimator
//! takes the median chroma over bright, non-clipped pixels; without a
//! near-neutral ("gray pixel") gate, a dominant yellow region pulls the median
//! toward its own low blue/green ratio, so the blue gain is over-boosted and
//! genuinely neutral objects render blue.

// The locus chroma constants are copied verbatim from the tuning (CCM_LOCUS[0]);
// keep their full precision as the tuning module does.
#![allow(clippy::excessive_precision)]

use gc2607_isp::pipeline::{self, Planes};

/// Warm-light white locus vertex in raw chroma (r/g, b/g): CCM_LOCUS[0]
/// (2963 K). A neutral surface lit by this illuminant has exactly this chroma.
const NEUTRAL_RG: f32 = 0.717_171_729;
const NEUTRAL_BG: f32 = 0.371_717_215;

/// Build a half-res GRBG scene: `frac_neutral` of the pixels are a bright
/// neutral surface at the warm locus chroma; the rest are a saturated yellow
/// object (high R/G, very low B). All values stay within the linear range and
/// below the AWB clip threshold, so the only thing separating the two
/// populations is chroma, not brightness or clipping.
fn yellow_dominated_scene(ww: usize, hh: usize, frac_neutral_tenths: usize) -> Planes {
    let n = ww * hh;
    let (mut gr, mut r, mut b, mut gb) =
        (vec![0f32; n], vec![0f32; n], vec![0f32; n], vec![0f32; n]);
    for i in 0..n {
        let neutral = i % 10 < frac_neutral_tenths;
        // Neutral patch is the brighter surface (g = 700) so it is never
        // excluded by the green-brightness mask; yellow is dimmer (g = 600).
        let (g, rg, bg) = if neutral {
            (700.0f32, NEUTRAL_RG, NEUTRAL_BG)
        } else {
            (600.0f32, 1.0f32, 0.185f32) // saturated yellow, far off the locus
        };
        gr[i] = g;
        gb[i] = g;
        r[i] = rg * g;
        b[i] = bg * g;
    }
    Planes {
        hh,
        ww,
        gr,
        r,
        b,
        gb,
    }
}

#[test]
fn awb_ignores_dominant_saturated_object() {
    // 30% neutral, 70% saturated yellow: the yellow is the majority of bright
    // pixels, so an ungated median lands on the yellow chroma.
    let planes = yellow_dominated_scene(120, 100, 3);
    let est = pipeline::estimate(&planes);

    // Correct white balance neutralises the neutral surface: its raw chroma is
    // (NEUTRAL_RG, NEUTRAL_BG), so the gains must be the reciprocals.
    let want_r = 1.0 / NEUTRAL_RG as f64; // ~1.394
    let want_b = 1.0 / NEUTRAL_BG as f64; // ~2.690

    assert!(
        (est.gains[0] - want_r).abs() < 0.05,
        "red gain {:.3} should match the neutral surface ({want_r:.3}); a \
         dominant yellow object must not bias it",
        est.gains[0]
    );
    assert_eq!(est.gains[1], 1.0, "green gain is the reference");
    assert!(
        (est.gains[2] - want_b).abs() < 0.10,
        "blue gain {:.3} should match the neutral surface ({want_b:.3}); the \
         reported blue cast is this gain being over-boosted toward the yellow \
         object's low b/g",
        est.gains[2]
    );
}
