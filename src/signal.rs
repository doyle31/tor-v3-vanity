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
//!
//! # CUDA undoes the installation
//!
//! Installing these handlers from `main` is necessary and not sufficient. The
//! CUDA driver resets SIGTERM and SIGINT to the default while it initialises,
//! and does it *nondeterministically*: two runs of the same binary, sampled at
//! the same point six seconds in, showed `/proc/self/status` `SigCgt` of
//! `0x100004442` (both caught) and `0x10000` (neither). `sigaction` returning
//! success proves nothing, because it does -- the disposition is simply gone by
//! the time anyone looks.
//!
//! So `gpu` re-installs from the worker thread once the context exists, and
//! re-asserts on every launch. The symptom when this is missing is that Ctrl-C
//! and `kill` terminate the process outright, so none of the shutdown above ever
//! runs and a `.json` array file is left without its closing bracket -- which is
//! the exact failure this module documents as its reason to exist.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// Set by the signal handler, polled by the main loop.
static SIGNALLED: AtomicBool = AtomicBool::new(false);

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
        //
        // The cast goes through an explicit `extern "C" fn` pointer type on
        // purpose: casting the function item straight to `usize` compiles, but
        // rustc warns about it because it is not a documented conversion. Going
        // via the pointer type is the sanctioned spelling and says what the
        // value actually is.
        unsafe {
            let mut sa: libc::sigaction = std::mem::zeroed();
            sa.sa_sigaction = on_signal as extern "C" fn(libc::c_int) as usize;
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

    /// The mask of signals the kernel reports as caught, from
    /// `/proc/self/status`.
    ///
    /// `sigaction` returning success is *not* evidence that a handler was
    /// installed. A wrongly laid-out `sigaction` writes the handler somewhere the
    /// kernel does not read, leaves the disposition at SIG_DFL, and still
    /// returns 0. This reads back what the kernel actually believes, which is the
    /// only way to catch that class of bug.
    ///
    /// It also catches the bug this module actually had: creating a CUDA context
    /// resets SIGTERM and SIGINT, so a handler installed before the context
    /// exists is silently discarded. No test in a process without a CUDA context
    /// can see that, so it is covered by `gpu` re-installing after context
    /// creation -- and this helper is what proved the re-install was needed.
    fn caught_mask() -> u64 {
        let status = std::fs::read_to_string("/proc/self/status").expect("/proc/self/status");
        for line in status.lines() {
            if let Some(rest) = line.strip_prefix("SigCgt:") {
                return u64::from_str_radix(rest.trim(), 16).expect("hex mask");
            }
        }
        panic!("no SigCgt in /proc/self/status");
    }

    #[test]
    fn install_actually_catches_sigterm_and_sigint() {
        // Assert the absolute property, not a delta from before install(). An
        // earlier version compared the two, on the theory that the harness might
        // have installed handlers of its own. That made the test
        // order-dependent: `install_is_idempotent_in_practice` installs them
        // too, so under the default parallel test runner the "before" mask
        // already had both bits and install() legitimately changed nothing.
        //
        // The absolute check is the one that matters. If the `sigaction` call
        // were malformed, or wrote the handler where the kernel does not read it,
        // the bits would simply be absent and this fails.
        install().expect("sigaction");
        let mask = caught_mask();
        for sig in [libc::SIGINT, libc::SIGTERM] {
            assert!(
                mask >> (sig - 1) & 1 == 1,
                "signal {sig} is not caught after install() (SigCgt={mask:#x}); the \
                 disposition is still SIG_DFL, so the process would be killed \
                 instead of shutting down cleanly"
            );
        }
    }

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
