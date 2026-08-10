//! Retry policy for a camera session that stopped delivering frames.
//!
//! A capture stall (no completed request within `camera::RECV_TIMEOUT`) or a
//! buffer that cannot be requeued leaves the libcamera session unusable while
//! the daemon itself is healthy: dropping the session and opening a fresh one
//! restores the stream. A system suspend/resume cycle produces exactly this —
//! the session that spanned the sleep never delivers another frame, but a new
//! session works.
//!
//! Retrying unconditionally would spin for as long as the camera is genuinely
//! unavailable, so the attempts are spaced by a doubling backoff and bounded.
//! Once the bound is reached the daemon stops, leaving a restart (and the
//! failure record) to the service manager.

use std::time::Duration;

/// Consecutive failed attempts after which recovery is abandoned.
pub const MAX_ATTEMPTS: u32 = 5;

/// Delay before the second attempt; doubled for each further consecutive
/// failure. The first attempt is immediate: a stale session after resume is
/// usually replaced successfully on the first try.
pub const BASE_BACKOFF: Duration = Duration::from_secs(1);

/// Upper bound on the delay between attempts.
pub const MAX_BACKOFF: Duration = Duration::from_secs(8);

/// What to do after a camera session was lost.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    /// Wait this long, then open a fresh session. The caller keeps the output
    /// device fed while waiting so a consumer sees a paused stream rather than
    /// a dead device.
    Retry(Duration),
    /// Too many consecutive failures: stop instead of retrying forever.
    GiveUp,
}

/// Decide what to do after the `attempt`-th consecutive lost session, counting
/// from 1 for the first loss since a session that did deliver frames.
///
/// Delays: none, then 1 s, 2 s, 4 s, 8 s, capped at [`MAX_BACKOFF`].
pub fn action(attempt: u32) -> Action {
    let attempt = attempt.max(1);
    if attempt > MAX_ATTEMPTS {
        return Action::GiveUp;
    }
    if attempt == 1 {
        return Action::Retry(Duration::ZERO);
    }
    // Shift by (attempt - 2) so the second attempt waits BASE_BACKOFF. The
    // shift is bounded well below u32's width by MAX_ATTEMPTS, and the delay is
    // capped anyway.
    let delay = BASE_BACKOFF.saturating_mul(1u32 << (attempt - 2).min(16));
    Action::Retry(delay.min(MAX_BACKOFF))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_attempt_is_immediate() {
        assert_eq!(action(1), Action::Retry(Duration::ZERO));
        // A caller that mislabels the first loss as attempt 0 still retries.
        assert_eq!(action(0), Action::Retry(Duration::ZERO));
    }

    #[test]
    fn backoff_doubles_and_is_capped() {
        assert_eq!(action(2), Action::Retry(Duration::from_secs(1)));
        assert_eq!(action(3), Action::Retry(Duration::from_secs(2)));
        assert_eq!(action(4), Action::Retry(Duration::from_secs(4)));
        assert_eq!(action(5), Action::Retry(Duration::from_secs(8)));
        for attempt in 1..=MAX_ATTEMPTS {
            match action(attempt) {
                Action::Retry(d) => assert!(d <= MAX_BACKOFF, "attempt {attempt}: {d:?}"),
                Action::GiveUp => panic!("attempt {attempt} within the bound must retry"),
            }
        }
    }

    #[test]
    fn gives_up_past_the_bound() {
        assert_eq!(action(MAX_ATTEMPTS + 1), Action::GiveUp);
        assert_eq!(action(u32::MAX), Action::GiveUp);
    }
}
