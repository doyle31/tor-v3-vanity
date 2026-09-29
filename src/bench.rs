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
) -> Option<(u64, f64)> {
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
    let (_dropped, workers) =
        gpu::start(prefixes, send, send_tries, Some(stop.clone()), launch).unwrap();

    let start = Instant::now();
    // Interruptible, so Ctrl-C does not park the process here for the full
    // duration; see `signal::sleep_unless_signalled`. Elapsed time is still
    // measured with `Instant`, so slicing the wait costs no accuracy.
    let completed = crate::signal::sleep_unless_signalled(Duration::from_secs(secs));
    let elapsed = start.elapsed().as_secs_f64();
    let keys = total.load(Ordering::Relaxed);

    // Wait for the launch loop to finish its in-flight launch and exit, so the
    // next rep starts on an idle GPU. Joining is exact; this used to be a
    // 500 ms sleep, which was a guess at how long a launch takes and left the
    // next rep measuring a GPU that was still busy when the sleep ran out.
    stop.store(true, Ordering::Relaxed);
    for worker in workers {
        let _ = worker.join();
    }
    if completed {
        Some((keys, elapsed))
    } else {
        None
    }
}

/// Run `reps` timed runs and return the best keys/s, in millions.
///
/// `None` means a run was interrupted, so there is no valid rate to report.
pub fn kernel(prefixes: &[String], secs: u64, reps: usize, launch: Launch) -> Option<f64> {
    let mut best = 0f64;
    for rep in 1..=reps {
        let Some((keys, elapsed)) = once(prefixes, secs, launch) else {
            eprintln!("interrupted: no benchmark result");
            return None;
        };
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
    Some(best / 1e6)
}

/// A random base32 prefix of `len` characters.
///
/// Used when `--bench` is given no prefix, so each repetition measures a
/// different one. That is deliberate: a single fixed prefix can be unlucky, and
/// a benchmark that reports the best of three identical runs is not measuring the
/// thing the tool does.
pub fn random_prefix(len: usize) -> String {
    const ALPHABET: &[u8] = b"abcdefghijklmnopqrstuvwxyz234567";
    let mut seed = [0u8; 4];
    rand::rng().fill_bytes(&mut seed);
    let mut x = u32::from_le_bytes(seed);
    (0..len)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            ALPHABET[(x >> 16) as usize % ALPHABET.len()] as char
        })
        .collect()
}

/// Run whichever benchmark `matching` selects.
///
/// The two numbers bracket the truth and mean different things, so they are
/// labelled rather than conflated:
///
/// * **matching on** runs a real search in a child process and counts the keys it
///   writes, so the rate is *found* per second -- the number that predicts how
///   long a search takes. The examined rate is then *derived* by multiplying by
///   `2^(5*len)`, which is an expectation, not a measurement: a real search's
///   yield is geometric, so a short run can land well above or below it.
/// * **matching off** times the kernel in-process and counts launches, giving
///   *examined* per second directly, with no matching or host work at all.
pub fn run(
    prefixes: &[String],
    secs: u64,
    reps: usize,
    matching: bool,
    launch: Launch,
    verbose: bool,
) -> Option<f64> {
    if matching {
        prefixes_bench(prefixes, secs, reps, launch, verbose)
    } else {
        kernel(&[], secs, reps, launch)
    }
}

/// Benchmark end-to-end against `prefixes`, reporting keys *found* per second.
///
/// With no prefixes, a random one of `RANDOM_BENCH_PREFIX_CHARS` characters is
/// generated per repetition.
///
/// Exercises the whole path: capture ring, readback, host-side key derivation,
/// prefix verification and the output sink.
fn prefixes_bench(
    prefixes: &[String],
    secs: u64,
    reps: usize,
    launch: Launch,
    verbose: bool,
) -> Option<f64> {
    let mut best = 0f64;

    for rep in 1..=reps {
        let prefixes = if prefixes.is_empty() {
            vec![random_prefix(crate::cli::random_bench_prefix_chars())]
        } else {
            prefixes.to_vec()
        };
        // One rate for all of them, so a multi-prefix run is compared like for
        // like against a single-prefix one. Several prefixes of equal length each
        // yield one key per 2^(5*len), so the combined expectation is
        // len * prefixes.len(); the run below divides by that.
        let longest = prefixes.iter().map(|p| p.len()).max().unwrap_or(1);
        let uniform = prefixes.iter().all(|p| p.len() == longest);
        assert!(
            uniform,
            "--bench prefixes must all be the same length, so the combined rate \
             is meaningful; got {:?}",
            prefixes
        );
        let expected = 2f64.powi(5 * longest as i32) * prefixes.len() as f64;

        let dir = std::env::temp_dir().join(format!("t3v-bench-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let out = dir.join("keys.csv");
        let _ = std::fs::remove_file(&out);

        let start = Instant::now();
        let mut cmd = std::process::Command::new(std::env::current_exe().unwrap());
        cmd.arg("--log-interval").arg("0");
        for p in &prefixes {
            cmd.arg(p);
        }
        cmd.arg("-o").arg(&out);
        if verbose {
            cmd.arg("--verbose");
        }
        let _ = &launch; // geometry applies to the parent; the child re-derives it
        let mut child = cmd.spawn().expect("failed to launch search");

        // Let the child run, then read the file exactly once.
        //
        // This used to poll: sleep 500 ms, re-read the whole output file, count
        // the lines, repeat. At a 1- or 2-character prefix that is tens of MB/s
        // of reads *inside* the measured window, competing with the child for
        // page cache and CPU, so the reported rate was partly a measurement of
        // the harness. Nothing needs the intermediate samples -- the child is
        // killed at the end and the file on disk is the whole result -- so
        // sample once, after the kill.
        //
        // Interruptible, so Ctrl-C does not park the process here for the full
        // duration; see `signal::sleep_unless_signalled`. Elapsed time is still
        // measured with `Instant`, so slicing the wait costs no accuracy.
        let completed = crate::signal::sleep_unless_signalled(Duration::from_secs(secs));
        let elapsed = start.elapsed().as_secs_f64();
        // `kill` is a no-op on an already-exited child and `wait` reaps it, so
        // this is correct on both paths. It is also the *only* thing that reaps
        // it: the child gets the same SIGINT the parent does, so on Ctrl-C it
        // exits first and would otherwise sit as an unreaped zombie for the
        // rest of the run.
        let _ = child.kill();
        let _ = child.wait();

        if !completed {
            // The child was stopped by the signal part-way through, so counting
            // its file over a window it did not run for would report a rate far
            // below the real one. Say nothing rather than say something false.
            eprintln!("interrupted: discarding partial benchmark");
            let _ = std::fs::remove_dir_all(&dir);
            return None;
        }

        // Counting the file is the ground truth: it is what a real search
        // produced, including anything the host had to verify and discard.
        let counted = std::fs::read_to_string(&out)
            .map(|s| s.lines().filter(|l| !l.starts_with("onion,")).count() as u64)
            .unwrap_or(0);

        let rate = counted as f64 / elapsed;
        best = best.max(rate);
        if reps > 1 {
            println!(
                "  run {}/{}: {} -> {:.2} keys found/s (implies {:.0} keys/s examined)",
                rep,
                reps,
                prefixes.join(","),
                rate,
                rate * expected
            );
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
    Some(best)
}

/// Tests for the `--bench` plumbing: the random prefix it generates, and the
/// rules that decide which measurement runs.
#[cfg(test)]
mod bench_tests {
    use super::*;

    #[test]
    fn a_random_prefix_is_the_requested_length_and_legal_base32() {
        for len in 1..=12usize {
            let p = random_prefix(len);
            assert_eq!(p.chars().count(), len, "length for {len}");
            assert!(
                p.chars()
                    .all(|c| c.is_ascii_lowercase() || ('2'..='7').contains(&c)),
                "{p:?} must be base32 (a-z, 2-7)"
            );
        }
    }

    #[test]
    fn random_prefixes_actually_vary() {
        // The point of generating one per repetition: a fixed prefix can be
        // unlucky, and a benchmark reporting the best of three identical runs is
        // not measuring what the tool does. 12 four-character prefixes colliding
        // is about 1 in 4^12, so this cannot flake in practice.
        let seen: std::collections::HashSet<String> = (0..12)
            .map(|_| random_prefix(4))
            .collect();
        assert!(
            seen.len() > 1,
            "expected variety across 12 draws, got {:?}",
            seen
        );
    }

    #[test]
    fn the_default_bench_prefix_length_is_four() {
        // Load-bearing for the reported rate, and the reason is in
        // `cli::RANDOM_BENCH_PREFIX_CHARS`: at one character a match in 32 costs
        // about 22% of the rate to ring publishing, so `--bench` would report a
        // figure for a search nobody runs.
        assert_eq!(crate::cli::random_bench_prefix_chars(), 4);
    }
}
