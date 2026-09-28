//! tor-v3-vanity: find a Tor v3 hidden service address with a chosen prefix.
//!
//! Layout:
//! * [`cli`] parses arguments into a resolved [`cli::Options`], once.
//! * [`gpu`] owns CUDA setup, the launch loop and the capture-ring readback.
//! * [`bench`] times the kernel and the end-to-end pipeline.
//! * [`output`] decides where matched keypairs go.
//! * `main` wires them together and runs the key-emission loop.

use std::time::Duration;
use std::time::Instant;

use sha3::{Digest, Sha3_256};
use tor_v3_vanity_core as core;

pub mod bench;
pub mod cli;
pub mod gpu;
pub mod output;
pub mod signal;

use cli::Mode;

pub struct Pubkey(pub [u8; 32]);

impl AsRef<[u8; 32]> for Pubkey {
    fn as_ref(&self) -> &[u8; 32] {
        &self.0
    }
}

pub struct PrettyDur(chrono::Duration);

impl std::fmt::Display for PrettyDur {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.0.num_weeks() >= 52 {
            write!(f, "{} years, ", self.0.num_weeks() / 52)?;
        }
        if self.0.num_weeks() % 52 > 0 {
            write!(f, "{} weeks, ", self.0.num_weeks() % 52)?;
        }
        if self.0.num_days() % 7 > 0 {
            write!(f, "{} days, ", self.0.num_days() % 7)?;
        }
        if self.0.num_hours() % 24 > 0 {
            write!(f, "{} hours, ", self.0.num_hours() % 24)?;
        }
        if self.0.num_minutes() % 60 > 0 {
            write!(f, "{} minutes, ", self.0.num_minutes() % 60)?;
        }
        write!(f, "{} seconds", self.0.num_seconds() % 60)
    }
}

pub fn pubkey_to_onion(pubkey: &[u8; 32]) -> String {
    let mut hasher = Sha3_256::new();
    hasher.update(b".onion checksum");
    hasher.update(pubkey);
    hasher.update(&[3]);
    let mut onion = [0; 35];
    onion[..32].clone_from_slice(pubkey);
    onion[32..34].clone_from_slice(&hasher.finalize()[..2]);
    onion[34] = 3;
    format!(
        "{}.onion",
        base32::encode(base32::Alphabet::RFC4648 { padding: false }, &onion).to_lowercase()
    )
}

/// Seconds to a `chrono::Duration`, saturating instead of overflowing.
///
/// The expected duration is `elapsed / (tried / 2^(5*len))`, which for a long
/// prefix exceeds what `chrono` can represent -- `from_std` returns `None`
/// there, and the old `.unwrap()` killed the progress thread outright. A
/// 20-character prefix needs 2^100 keys, so the estimate is around 10^22
/// seconds and any real run hit this.
fn to_chrono(secs: f64) -> chrono::Duration {
    const MAX_SECS: f64 = 9.0e9; // ~285 years, inside chrono's range
    chrono::Duration::from_std(Duration::from_secs_f64(secs.clamp(0.0, MAX_SECS)))
        .unwrap_or_else(|_| chrono::Duration::max_value())
}

/// Reject bad build-time tunables here, on the main thread, with a message
/// that names the variable.
///
/// The device crate asserts the same conditions, but it runs inside a worker
/// thread: a failure there panicked that thread, `main` then died with an
/// unrelated `RecvError`, and the actual cause was buried. Worse, the user had
/// no way to tell which of several `T3V_*` variables was at fault.
fn validate_tunables() {
    let mut bad = |name: &str, value: usize, ok: bool, want: &str| {
        if !ok {
            eprintln!(
                "error: {} is {}, but {}. Rebuild without that variable, or set it to {}.",
                name, value, want, name
            );
            std::process::exit(2);
        }
    };
    bad(
        "T3V_CHUNK",
        core::CHUNK,
        core::CHUNK > 0 && core::BATCH % core::CHUNK == 0,
        "it must be non-zero and divide T3V_BATCH",
    );
    bad(
        "T3V_BATCH",
        core::BATCH,
        core::BATCH > 0 && core::BATCH % core::CHUNK == 0,
        "it must be non-zero and be a multiple of T3V_CHUNK",
    );
    // `build_comb_table` requires 2..=4. 5 is the dangerous one: 252 % 5 == 2,
    // so a 5-bit window silently leaves the top two scalar bits unread, and
    // clamping sets one of them on every key.
    bad(
        "T3V_COMB_W",
        core::group::COMB_W,
        (2..=4).contains(&core::group::COMB_W) && core::group::COMB_BITS % core::group::COMB_W == 0,
        "it must be 2, 3 or 4, and must divide 252",
    );
}

fn main() {
    let opts = cli::parse();
    // Install before anything can block, so a signal during startup is still
    // noticed. Failure is not fatal: a search without it still works, it just
    // will not close its output file cleanly.
    if let Err(e) = signal::install() {
        eprintln!("warning: cannot install signal handlers: {}", e);
    }
    validate_tunables();
    let launch = gpu::Launch {
        verbose: opts.verbose,
        threads: opts.threads,
        blocks_per_sm: opts.blocks_per_sm,
    };

    match opts.mode {
        Mode::Bench {
            secs,
            reps,
            prefixes,
        } => {
            let m = bench::kernel(&prefixes, secs, reps, launch);
            println!("{:.3}", m);
        }
        Mode::BenchPrefix { len, secs, reps } => {
            let rate = bench::prefix_len(len, secs, reps, opts.verbose);
            println!("{:.3}", rate);
            eprintln!("prefix length {}: {:.3} keys found/s", len, rate);
        }
        Mode::Search => {
            search(&opts, launch);
        }
    }
}

fn search(opts: &cli::Options, launch: gpu::Launch) {
    let prefixes = &opts.prefixes;
    let max_len = prefixes
        .iter()
        .fold(0, |acc, x| std::cmp::max(acc, x.len()));

    // One of two sinks: a single consolidated file, or a directory of per-key
    // Tor key files.
    let mut sink = match &opts.out {
        Some(path) => {
            let chosen = opts
                .format
                .unwrap_or_else(|| output::Format::from_path(path));
            eprintln!(
                "Writing matches to {} ({}, {})",
                path.display(),
                chosen.name(),
                match opts.durability {
                    output::Durability::Durable => "fsync per key",
                    output::Durability::Buffered => "buffered",
                }
            );
            output::Sink::open_format(path, chosen, opts.durability).unwrap()
        }
        None => {
            let dst = opts
                .dst
                .clone()
                .unwrap_or_else(|| std::env::current_dir().unwrap());
            assert!(dst.is_dir(), "dst must be a directory");
            output::Sink::dir(&dst)
        }
    };

    let (send, recv) = crossbeam_channel::unbounded();
    let (send_tries, recv_tries) = crossbeam_channel::bounded(1);
    // A shutdown request has to reach the GPU threads, so they get the stop flag
    // and this loop watches the signal flag.
    let stop = signal::stopper();
    let dropped_keys =
        gpu::start(prefixes, send, send_tries, Some(stop.clone()), launch).unwrap();

    let log_interval = opts.log_interval;
    std::thread::spawn(move || {
        let now = Instant::now();
        let mut last_log = Instant::now();
        let mut tries = 0_f64;
        let expected = 2_f64.powi(5 * max_len as i32);
        loop {
            match recv_tries.recv() {
                Ok(n) => tries += n as f64,
                // The search threads are gone; nothing left to report on.
                Err(_) => return,
            }
            let dur = now.elapsed().as_secs_f64();
            let progress = tries / expected;
            if log_interval > Duration::from_secs(0) && last_log.elapsed() > log_interval {
                println!("Tried {:.0} / {:.0} (expected) keys.", tries, expected);
                println!(
                    "Running for {} / {} (expected).",
                    PrettyDur(to_chrono(dur)),
                    PrettyDur(to_chrono(dur / progress))
                );
                let dropped = dropped_keys.load(std::sync::atomic::Ordering::Relaxed);
                if dropped > 0 {
                    println!(
                        "WARNING: {} keys matched but were dropped -- a launch published \
                         more than the {} slots per prefix.",
                        dropped,
                        core::SLOTS
                    );
                }
                last_log = Instant::now();
            }
        }
    });

    loop {
        // Check the signal on *every* iteration, not only when the channel is
        // idle. A busy search delivers matches faster than they can be drained,
        // so `recv_timeout` never fires and a check that lives on the timeout
        // path alone is never reached -- the process then ignores SIGTERM
        // entirely. Polling is still needed so a signal is noticed while idle.
        if signal::requested().load(std::sync::atomic::Ordering::SeqCst) {
            break;
        }
        // The kernel returns the 64-byte expanded key, so the pubkey comes
        // straight from it with no host-side key derivation.
        let expanded = match recv.recv_timeout(Duration::from_millis(200)) {
            Ok(k) => k,
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => continue,
            // Every search thread is gone; nothing more can arrive.
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => break,
        };
        let esk = ed25519_dalek::ExpandedSecretKey::from_bytes(&expanded[..]).unwrap();
        let pk: ed25519_dalek::PublicKey = (&esk).into();
        let onion = pubkey_to_onion(pk.as_bytes());

        // Several threads can match the same prefix in one launch, and they all
        // write to that prefix's single `out` slot, so the key read back may not
        // be the one that matched. Verify before emitting it; a non-match just
        // means we lost the race, and the search continues.
        let matched = match prefixes.iter().find(|p| onion.starts_with(p.as_str())) {
            Some(p) => p.clone(),
            None => continue,
        };

        println!("{}", onion);
        sink.write(&output::Key {
            onion: onion.clone(),
            expanded: esk.to_bytes(),
            matched,
        })
        .unwrap();
    }

    // Shut down cleanly. Stopping the GPU threads first stops new keys arriving
    // while the sink is being closed.
    stop.store(true, std::sync::atomic::Ordering::Relaxed);
    // `finish` is what closes a `.json` array with its `]`; without it such a
    // file is unparseable, and every other format is left with a torn last
    // record. It is also safe to call for CSV and JSON Lines, where it is just
    // a flush.
    if let Err(e) = sink.finish() {
        eprintln!("error closing output: {}", e);
    }
}
