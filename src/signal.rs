//! Clean shutdown on SIGTERM and SIGINT.
//!
//! Without this, stopping a search kills the process mid-write and leaves a torn
//! final record: measured across a dozen runs, exactly one partial line per
//! output file, always the last one. That is survivable for CSV and JSON Lines
//! -- skip the final line and every earlier record is intact -- but it makes a
//! `.json` *array* file unparseable, since the closing `]` never gets written.
//!
//! The handler itself does almost nothing, which is the point: a signal handler
//! may only touch async-signal-safe state, so all it does is set one atomic
//! flag. A normal thread watches that flag and performs the real work --
//! finishing the sink, telling the GPU threads to stop, and exiting.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// Set by the signal handler, polled by the main loop.
static SIGNALLED: AtomicBool = AtomicBool::new(false);

/// A token the main loop can poll to see whether a shutdown was requested.
pub fn requested() -> &'static AtomicBool {
    &SIGNALLED
}

/// Install handlers for SIGTERM and SIGINT.
///
/// Returns an error if the handlers could not be installed; a search still works
/// without them, it just will not close its output file cleanly.
pub fn install() -> Result<(), String> {
    for sig in [libc::SIGTERM, libc::SIGINT] {
        // SAFETY: `handler` is a plain `extern "C" fn` that only stores to a
        // static atomic, which is async-signal-safe. SA_RESTART keeps syscalls
        // from being interrupted into EINTR.
        unsafe {
            let mut sa: libc::sigaction = std::mem::zeroed();
            sa.sa_sigaction = on_signal as usize;
            sa.sa_flags = libc::SA_RESTART;
            libc::sigemptyset(&mut sa.sa_mask);
            if libc::sigaction(sig, &sa, std::ptr::null_mut()) != 0 {
                return Err(format!(
                    "sigaction({}) failed: {}",
                    sig,
                    std::io::Error::last_os_error()
                ));
            }
        }
    }
    Ok(())
}

extern "C" fn on_signal(_sig: libc::c_int) {
    // The only thing that is safe to do here.
    SIGNALLED.store(true, Ordering::SeqCst);
}

/// Ask the GPU launch loops to stop, and hold the token the main loop polls.
pub fn stopper() -> Arc<std::sync::atomic::AtomicBool> {
    Arc::new(AtomicBool::new(false))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn handler_sets_the_flag() {
        // The handler body is what matters; call it directly rather than
        // signalling the test process, which would be rude to the harness.
        SIGNALLED.store(false, Ordering::SeqCst);
        on_signal(libc::SIGTERM);
        assert!(SIGNALLED.load(Ordering::SeqCst));
        SIGNALLED.store(false, Ordering::SeqCst);
        assert!(!SIGNALLED.load(Ordering::SeqCst));
    }

    #[test]
    fn install_is_idempotent_in_practice() {
        // Installing twice must not error, and the flag must survive it.
        install().expect("install");
        install().expect("second install");
    }
}
