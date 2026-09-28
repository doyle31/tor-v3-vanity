//! tor-v3-vanity: find a Tor v3 hidden service address with a chosen prefix.
//!
//! Layout:
//! * [`cli`] parses arguments into a resolved [`cli::Options`], once.
//! * [`gpu`] owns CUDA setup, the launch loop and the capture-ring readback.
//! * [`bench`] times the kernel and the end-to-end pipeline.
//! * [`output`] decides where matched keypairs go.
//! * `main` wires them together and runs the key-emission loop.

use std::io::Write;
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
    // `Rfc4648Lower`, not `Rfc4648`: base32 0.5 split the alphabet by case, so
    // `Rfc4648` now emits uppercase and `Rfc4648Lower` emits lowercase. A Tor
    // address is canonically lowercase, and the old code got there by encoding
    // uppercase and then calling `.to_lowercase()` -- a second full pass, and a
    // second String allocation, over every address formatted. Asking for the
    // lowercase alphabet directly produces the right bytes the first time.
    //
    // This is on the per-match path, so the saved allocation is per key found,
    // not per key examined.
    format!(
        "{}.onion",
        base32::encode(base32::Alphabet::Rfc4648Lower { padding: false }, &onion)
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
        .unwrap_or_else(|_| chrono::Duration::MAX)
}

/// Reject bad build-time tunables here, on the main thread, with a message
/// that names the variable.
///
/// The device crate asserts the same conditions, but it runs inside a worker
/// thread: a failure there panicked that thread, `main` then died with an
/// unrelated `RecvError`, and the actual cause was buried. Worse, the user had
/// no way to tell which of several `T3V_*` variables was at fault.
fn validate_tunables() {
    let bad = |name: &str, value: usize, ok: bool, want: &str| {
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
    //
    // This is not sufficient on its own. Creating the CUDA context resets
    // SIGTERM and SIGINT to the default, so `gpu` re-installs the handlers from
    // the worker thread once the context exists. See the comment there.
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
            matching,
        } => {
            // Two numbers, and they mean different things, so each is labelled
            // with the unit it is in rather than left for the reader to infer.
            //
            // `run` reports keys *found*/s when matching and keys *examined*/s
            // when not; printing the wrong label there would make a rate look
            // 2^20 times better or worse than it is.
            let result = bench::run(&prefixes, secs, reps, matching, launch, opts.verbose);
            match result {
                // `None` means the run was interrupted, and an interrupted
                // benchmark has no valid rate: print nothing rather than print a
                // number measured over a window that was cut short.
                None => {}
                Some(found) if matching => {
                    println!("{:.3}", found);
                    let longest = prefixes
                        .iter()
                        .map(|p| p.len())
                        .max()
                        .unwrap_or(crate::cli::random_bench_prefix_chars());
                    eprintln!(
                        "{:.3} keys found/s over {}-character prefix(es); \
                         implies {:.0} keys examined/s",
                        found,
                        longest,
                        found * 2f64.powi(5 * longest as i32) * prefixes.len().max(1) as f64,
                    );
                }
                Some(examined) => {
                    println!("{:.3}", examined);
                    eprintln!(
                        "{:.3} keys examined/s (kernel only, no matching -- the \
                         ceiling, not what a search achieves)",
                        examined
                    );
                }
            }
        }
        Mode::Search => {
            search(&opts, launch);
        }
    }
}

/// Report an output failure once, then stop the search rather than losing
/// keys silently. A closed pipe is the normal way a piped run ends; anything
/// else is a real problem worth surfacing loudly.
fn report_write_failure(what: &str, e: &std::io::Error, written: u64) {
    if e.kind() == std::io::ErrorKind::BrokenPipe {
        eprintln!("the reader closed the pipe; stopping after {written} keys");
    } else {
        eprintln!("error: writing {what} failed: {e}");
        eprintln!("       stopping after {written} keys, rather than losing more silently");
    }
}

fn search(opts: &cli::Options, launch: gpu::Launch) {
    // Reject prefixes the kernel cannot test, before any worker starts.
    //
    // `BytePrefixOwned::from_str` asserts on these, and an assert that fires
    // inside a worker thread tears down the *entire* search rather than
    // skipping the one unusable prefix -- so a single over-long word out of
    // a few hundred silently costs you all the others.
    //
    // The limit is not a policy choice. The kernel tests the low eight bytes
    // of the encoding in a single 64-bit word, so a prefix reaching past
    // byte 7 cannot be matched at all.
    let (usable, rejected): (Vec<String>, Vec<String>) = opts
        .prefixes
        .iter()
        .cloned()
        .partition(|p| p.len() <= core::BytePrefix::MAX_PREFIX_CHARS);
    if !rejected.is_empty() {
        eprintln!(
            "warning: {} prefix(es) longer than {} characters cannot be tested by the kernel \
             and were skipped: {}",
            rejected.len(),
            core::BytePrefix::MAX_PREFIX_CHARS,
            rejected.join(", ")
        );
    }
    if usable.is_empty() {
        eprintln!("error: no searchable prefixes remain");
        return;
    }
    let prefixes = &usable;
    let max_len = prefixes
        .iter()
        .fold(0, |acc, x| std::cmp::max(acc, x.len()));
    // `--count N` stops after N keys; 0 means run until interrupted.
    let limit = opts.count;

    // One of two sinks: a single consolidated file, or a directory of per-key
    // Tor key files.
    //
    // `consolidated` also decides where the human-facing lines go. When the
    // records themselves are on stdout -- `--out /dev/stdout` into a pipe --
    // the address line and the progress lines must not go there too: they
    // are written through a different handle with a different buffer, so the
    // two interleave and the stream is mostly garbage. Measured on a
    // two-character prefix, 400 lines of `--out /dev/stdout` were 337 bare
    // addresses and 63 CSV rows.
    let mut sink_is_consolidated = false;
    let mut sink = match &opts.out {
        Some(path) => {
            let chosen = opts
                .format
                .unwrap_or_else(|| output::Format::from_path(path));
            let existing = output::Existing::from_flags(opts.append, opts.overwrite);
            // Say what will happen to an existing file, because the difference is
            // destructive. With neither flag the sink refuses outright, and this
            // line is what tells the user why before they get the error.
            let exists = path.exists();
            let disposition = match (existing, exists) {
                (output::Existing::Append, true) => "appending to",
                (output::Existing::Append, false) => "appending to (new)",
                (output::Existing::Overwrite, true) => "OVERWRITING",
                (output::Existing::Overwrite, false) => "writing",
                (output::Existing::Refuse, true) => "refusing to overwrite",
                (output::Existing::Refuse, false) => "writing",
            };
            // fsync per key is impossible on a pipe or device -- it fails
            // with EINVAL, which used to surface as a panic on the first
            // match. Downgrade instead, so `--out /dev/stdout` just works.
            let mut durability = opts.durability;
            if durability.is_durable() && !output::Sink::is_regular_file(path) {
                eprintln!(
                    "note: {} is not a regular file, so it cannot be fsynced; using buffered output",
                    path.display()
                );
                durability = output::Durability::Buffered;
            }
            eprintln!(
                "Writing matches to {} ({}, {}, {})",
                path.display(),
                chosen.name(),
                disposition,
                match durability {
                    output::Durability::Durable => "fsync per key",
                    output::Durability::Buffered => "buffered",
                }
            );
            sink_is_consolidated = true;
            // A refusal is a user error, not a crash: report it and exit non-zero
            // with the remedy, rather than unwrapping into a panic.
            match output::Sink::open_format(path, chosen, durability, existing) {
                Ok(sink) => sink,
                Err(e) => {
                    eprintln!("error: cannot open {}: {e}", path.display());
                    std::process::exit(1);
                }
            }
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
    let (dropped_keys, workers) =
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
                eprintln!("Tried {:.0} / {:.0} (expected) keys.", tries, expected);
                eprintln!(
                    "Running for {} / {} (expected).",
                    PrettyDur(to_chrono(dur)),
                    PrettyDur(to_chrono(dur / progress))
                );
                let dropped = dropped_keys.load(std::sync::atomic::Ordering::Relaxed);
                if dropped > 0 {
                    eprintln!(
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

    let mut written: u64 = 0;
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
        // The kernel hands back an *expanded* key: `expanded[..32]` is already
        // `clamp(SHA-512(seed)[..32])` and `expanded[32..]` is the nonce prefix.
        // That is the input `ExpandedSecretKey` wants, not a seed.
        //
        // It matters which one is passed. `ExpandedSecretKey::from_bytes`
        // clamps and reduces the 32 bytes it is given and uses them directly as
        // the scalar. `SigningKey::from_bytes`, by contrast, treats its argument
        // as a *seed*: it runs SHA-512 over it first
        // (`impl From<&SecretKey> for ExpandedSecretKey`) and uses the digest
        // as the scalar. Feeding an already-expanded key to `SigningKey`
        // therefore hashes it a second time and derives a public key unrelated
        // to the one the kernel matched on. Nothing errors -- the wrong address
        // is computed, fails the prefix check below, and is silently discarded,
        // so a search finds almost nothing and reports no reason why. On a
        // 2-character prefix that measured 412 keys in 6s against 418460 for
        // the correct derivation, the residue being pure coincidence.
        let esk = ed25519_dalek::hazmat::ExpandedSecretKey::from_bytes(&expanded);
        let pk = ed25519_dalek::VerifyingKey::from(&esk);
        let onion = pubkey_to_onion(pk.as_bytes());

        // Several threads can match the same prefix in one launch, and they all
        // write to that prefix's single `out` slot, so the key read back may not
        // be the one that matched. Verify before emitting it; a non-match just
        // means we lost the race, and the search continues.
        let matched = match prefixes.iter().find(|p| onion.starts_with(p.as_str())) {
            Some(p) => p.clone(),
            None => continue,
        };

        // Both of these can fail once the reader has gone away. Neither is
        // worth losing an hours-long search over, and a bare `unwrap()` did
        // exactly that: a closed pipe, or a full disk, killed the run with a
        // panic and lost every key after it.
        // The address goes to stdout only when the records are going to
        // files; otherwise stdout is the data stream and this would corrupt
        // it, so it moves to stderr.
        let addr_result = if sink_is_consolidated {
            writeln!(std::io::stderr(), "{}", onion)
        } else {
            writeln!(std::io::stdout(), "{}", onion)
        };
        if let Err(e) = addr_result {
            report_write_failure("the address", &e, written);
            break;
        }
        if let Err(e) = sink.write(&output::Key {
            onion: onion.clone(),
            expanded,
            matched,
        }) {
            report_write_failure("a key", &e, written);
            break;
        }
        written += 1;

        // `--count` stops the search once enough keys are on disk. The exit is
        // through `break` rather than a flag so the normal shutdown path below
        // still runs: the sink is finished, so a `.json` array gets its closing
        // bracket, and the GPU threads are joined before CUDA is torn down.
        //
        // A key that arrived in the same batch as the one that hit the limit is
        // not written, so the count is an exact bound rather than an overshoot:
        // the batch is drained from the channel in order and the loop breaks the
        // moment the target is reached.
        if limit != 0 && written >= limit {
            break;
        }
    }

    // Shut down cleanly. Ask the GPU threads to stop, then *wait* for them
    // before touching the sink: a worker still inside `stream.synchronize`
    // when the process starts tearing CUDA down gets `Deinitialized` back and
    // dies, which used to leave the search silently producing nothing.
    stop.store(true, std::sync::atomic::Ordering::Relaxed);
    for worker in workers {
        if worker.join().is_err() {
            eprintln!("warning: a GPU worker thread panicked; some keys may be missing");
        }
    }
    // `finish` is what closes a `.json` array with its `]`; without it such a
    // file is unparseable, and every other format is left with a torn last
    // record. It is also safe to call for CSV and JSON Lines, where it is just
    // a flush.
    if let Err(e) = sink.finish() {
        // A reader that closed the pipe is a normal way for a piped run to
        // end, so it does not warrant an error line.
        if e.kind() != std::io::ErrorKind::BrokenPipe {
            eprintln!("error closing output: {e}");
        }
    }
}

#[cfg(test)]
mod key_derivation_tests {
    //! The host's public key must be the *same* public key the kernel matched
    //! on, or every match is discarded at the prefix check and a search reports
    //! nothing while appearing to run.
    //!
    //! This has already gone wrong once. `ed25519-dalek` 1.x took an expanded
    //! key, but 2.x's `SigningKey::from_bytes` takes a *seed* and SHA-512s it
    //! first. The obvious-looking migration -- pass the first 32 bytes of the
    //! expanded key, now called a `SigningKey` -- compiles, runs, and produces a
    //! different public key for every key, because those bytes are already the
    //! digest and get hashed again. It is silent: no panic, no error, just a
    //! search that finds ~1/1024th of what it should.
    //!
    //! The check below is end-to-end on purpose. It derives the key exactly the
    //! way `search` does, then asserts the address the program would print
    //! against a value derived independently from RFC 8032, and separately
    //! against what the device crate's own field arithmetic produces. A test
    //! that only compared two host-side calls would pass under the broken
    //! migration, since both would be wrong together.

    use super::pubkey_to_onion;
    use tor_v3_vanity_core as core;

    /// RFC 8032 section 7.1, test vector 1.
    const SEED: [u8; 32] = [
        0x9d, 0x61, 0xb1, 0x9d, 0xef, 0xfd, 0x5a, 0x60, 0xba, 0x84, 0x4a, 0xf4, 0x92, 0xec, 0x2c,
        0xc4, 0x44, 0x49, 0xc5, 0x69, 0x7b, 0x32, 0x69, 0x19, 0x70, 0x3b, 0xac, 0x03, 0x1c, 0xae,
        0x7f, 0x60,
    ];
    const PUBLIC: [u8; 32] = [
        0xd7, 0x5a, 0x98, 0x01, 0x82, 0xb1, 0x0a, 0xb7, 0xd5, 0x4b, 0xfe, 0xd3, 0xc9, 0x64, 0x07,
        0x3a, 0x0e, 0xe1, 0x72, 0xf3, 0xda, 0xa6, 0x23, 0x25, 0xaf, 0x02, 0x1a, 0x68, 0xf7, 0x07,
        0x51, 0x1a,
    ];

    /// The expanded key for a seed: `SHA-512(seed)`, which is what the kernel
    /// derives and hands back.
    fn expand(seed: &[u8; 32]) -> [u8; 64] {
        core::sha512::sha512(seed)
    }

    /// Exactly what `search` does with a matched key.
    fn host_pubkey(expanded: &[u8; 64]) -> [u8; 32] {
        let esk = ed25519_dalek::hazmat::ExpandedSecretKey::from_bytes(expanded);
        *ed25519_dalek::VerifyingKey::from(&esk).as_bytes()
    }

    #[test]
    fn host_pubkey_matches_rfc8032() {
        let got = host_pubkey(&expand(&SEED));
        assert_eq!(
            got, PUBLIC,
            "host pubkey is not the RFC 8032 one; an expanded key is being \
             treated as a seed (or vice versa), so every match would be \
             discarded as a non-match at runtime"
        );
    }

    #[test]
    fn host_pubkey_matches_the_device_crate() {
        // The kernel does not call dalek. It clamps and multiplies with its own
        // field arithmetic, so agreement here is what makes the prefix check
        // meaningful: if these two disagree, the kernel matches one address and
        // the host prints another.
        let expanded = expand(&SEED);
        let mut scalar = [0u8; 32];
        scalar.copy_from_slice(&expanded[..32]);
        scalar[0] &= 248;
        scalar[31] &= 127;
        scalar[31] |= 64;

        let mut base: core::group::BaseTables = unsafe { std::mem::zeroed() };
        core::group::build_comb_table(&mut base.comb, &mut base.eight);
        let point = core::group::scalarmult_base_comb(&scalar, &base.comb);
        let zi = core::field::fe_invert(&point.z);
        let device_pubkey = core::group::ge_to_bytes_with_zi(&point, &zi);

        assert_eq!(
            host_pubkey(&expanded),
            device_pubkey,
            "host and device derive different public keys for the same expanded \
             key, so the host would reject every match the kernel publishes"
        );
    }

    #[test]
    fn onion_address_is_lowercase_base32() {
        // base32 0.5 made the alphabets case-specific. `Rfc4648` now emits
        // uppercase, and a Tor address that comes out uppercase is not
        // recognised as one, so this pins the lowercase alphabet.
        let onion = pubkey_to_onion(&PUBLIC);
        assert_eq!(onion.len(), 56 + ".onion".len());
        assert!(onion.ends_with(".onion"));
        assert!(
            onion[..56].bytes().all(|b| b.is_ascii_lowercase() || b.is_ascii_digit()),
            "onion body must be lowercase base32, got {:?}",
            onion
        );
        // And it must round-trip: decoding the body must give the pubkey back.
        let body = &onion[..56];
        let decoded = base32::decode(base32::Alphabet::Rfc4648Lower { padding: false }, body)
            .expect("onion body must be decodable");
        assert_eq!(&decoded[..32], &PUBLIC[..], "onion must encode the pubkey");
        assert_eq!(decoded[32..34].len(), 2);
        assert_eq!(decoded[34], 3, "version byte must be 3");
    }

    #[test]
    fn every_seed_agrees_between_host_and_device() {
        // One vector could be coincidence. Sweep deterministically so a
        // systematic disagreement -- an off-by-one in the clamp, a wrong half
        // taken of the digest -- shows up.
        let mut base: core::group::BaseTables = unsafe { std::mem::zeroed() };
        core::group::build_comb_table(&mut base.comb, &mut base.eight);
        for i in 0..64u8 {
            let mut seed = [0u8; 32];
            for (k, b) in seed.iter_mut().enumerate() {
                *b = i.wrapping_mul(31).wrapping_add(k as u8 * 7);
            }
            let expanded = expand(&seed);
            let mut scalar = [0u8; 32];
            scalar.copy_from_slice(&expanded[..32]);
            scalar[0] &= 248;
            scalar[31] &= 127;
            scalar[31] |= 64;
            let point = core::group::scalarmult_base_comb(&scalar, &base.comb);
            let zi = core::field::fe_invert(&point.z);
            assert_eq!(
                host_pubkey(&expanded),
                core::group::ge_to_bytes_with_zi(&point, &zi),
                "seed {} disagrees between host and device",
                i
            );
        }
    }
}
