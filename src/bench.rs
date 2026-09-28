//! Benchmarking.
//!
//! Two deliberately different measurements, because "how fast is this" has two
//! honest answers:
//!
//! * [`once`] times the kernel alone. Keys are counted on the host as launches
//!   complete, so nothing else is on the critical path: no readback, no key
//!   derivation, no file writes. With an empty prefix list the kernel also skips
//!   matching, so this measures pure key generation.
//! * [`prefix_len`] times the whole pipeline for a prefix of a given length and
//!   reports keys *found* per second, by counting the lines a real search
//!   actually wrote. That is the number that decides how long a search takes.
//!
//! The two differ by the entire host pipeline, so measuring one and quoting it
//! as the other is how a 1953x host-side bottleneck came to be reported as GPU
//! throughput.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use rand::RngCore;

use crate::gpu::{self, Launch};

/// Time the kernel for `secs`, returning `(keys examined, seconds)`.
///
/// Counting launches on the host is exact: each launch advances by a fixed
/// number of keys, so the total is a multiple of that with no sampling error.
pub fn once(
    prefixes: &[String],
    secs: u64,
    launch: Launch,
) -> (u64, f64) {
    use crossbeam_channel::bounded;

    // The counter channel is bounded(1), so it must be drained continuously or
    // the launch loop blocks on send and the measurement is of the channel
    // rather than of the GPU. A dedicated thread accumulates; this one only
    // samples the total.
    let (send_tries, recv_tries) = bounded::<u64>(1);
    let total = Arc::new(AtomicU64::new(0));
    let acc = total.clone();
    std::thread::spawn(move || {
        while let Ok(n) = recv_tries.recv() {
            acc.fetch_add(n, Ordering::Relaxed);
        }
    });

    let (send, recv) = bounded::<[u8; 64]>(4096);
    std::thread::spawn(move || while recv.recv().is_ok() {});

    // Each rep must fully stop the previous rep's launch loop, or two kernels
    // share the GPU and every run after the first is measured under contention.
    let stop = Arc::new(AtomicBool::new(false));
    gpu::start(prefixes, send, send_tries, Some(stop.clone()), launch).unwrap();

    let start = Instant::now();
    std::thread::sleep(Duration::from_secs(secs));
    let elapsed = start.elapsed().as_secs_f64();
    let keys = total.load(Ordering::Relaxed);

    stop.store(true, Ordering::Relaxed);
    // Let the launch loop finish its in-flight launch and exit before
    // returning, so the next rep starts on an idle GPU.
    std::thread::sleep(Duration::from_millis(500));
    (keys, elapsed)
}

/// Run `reps` timed runs and return the best keys/s, in millions.
pub fn kernel(prefixes: &[String], secs: u64, reps: usize, launch: Launch) -> f64 {
    let mut best = 0f64;
    for rep in 1..=reps {
        let (keys, elapsed) = once(prefixes, secs, launch);
        let rate = keys as f64 / elapsed;
        best = best.max(rate);
        if reps > 1 {
            println!(
                "  run {}/{}: {:.3}M keys/s ({} keys in {:.2}s)",
                rep,
                reps,
                rate / 1e6,
                keys,
                elapsed
            );
        }
    }
    best / 1e6
}

/// Benchmark end-to-end for a prefix of length `len`, reporting keys *found*
/// per second.
///
/// Exercises the whole path: capture ring, readback, host-side key derivation,
/// prefix verification and the output sink. A random prefix is used each rep so
/// one unlucky prefix cannot flatter or spoil the result; the expected yield is
/// one key per `2^(5*len)`.
pub fn prefix_len(len: usize, secs: u64, reps: usize, verbose: bool) -> f64 {
    const ALPHABET: &[u8] = b"abcdefghijklmnopqrstuvwxyz234567";
    let mut best = 0f64;

    for rep in 1..=reps {
        let mut seed = [0u8; 4];
        rand::thread_rng().fill_bytes(&mut seed);
        let mut x = u32::from_le_bytes(seed);
        let prefix: String = (0..len)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 17;
                x ^= x << 5;
                ALPHABET[(x >> 16) as usize % ALPHABET.len()] as char
            })
            .collect();

        let dir = std::env::temp_dir().join(format!("t3v-bench-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let out = dir.join("keys.csv");
        let _ = std::fs::remove_file(&out);

        let expected = 2f64.powi(5 * len as i32);
        let start = Instant::now();
        let mut cmd = std::process::Command::new(std::env::current_exe().unwrap());
        cmd.arg(&prefix).arg("-o").arg(&out).arg("--log-interval").arg("0");
        if verbose {
            cmd.arg("--verbose");
        }
        let mut child = cmd.spawn().expect("failed to launch search");

        // Let the child run, then read the file exactly once.
        //
        // This used to poll: sleep 500 ms, re-read the whole output file, count
        // the lines, repeat. At a 1- or 2-character prefix that is tens of MB/s
        // of reads *inside the measured window*, competing with the child for
        // page cache and CPU, so the reported rate was partly a measurement of
        // the harness. Nothing needs the intermediate samples -- the child is
        // killed at the end and the file on disk is the whole result -- so
        // sample once, after the kill.
        std::thread::sleep(Duration::from_secs(secs));
        let elapsed = start.elapsed().as_secs_f64();
        let _ = child.kill();
        let _ = child.wait();

        // Counting the file is the ground truth: it is what a real search
        // produced, including anything the host had to verify and discard.
        let counted = std::fs::read_to_string(&out)
            .map(|s| s.lines().filter(|l| !l.starts_with("onion,")).count() as u64)
            .unwrap_or(0);

        let rate = counted as f64 / elapsed;
        best = best.max(rate);
        if reps > 1 {
            println!(
                "  run {}/{}: prefix {} -> {:.2} keys found/s (implies {:.0} keys/s examined)",
                rep,
                reps,
                prefix,
                rate,
                rate * expected
            );
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
    best
}
