//! Command-line parsing.
//!
//! Every option is resolved here into a typed [`Options`], so the rest of the
//! program never touches `clap` and never has to re-derive a default. The rule
//! is that resolution happens exactly once, here, and everything downstream
//! reads a plain value.
//!
//! Two defaults are worth stating explicitly because they are the only places
//! where behaviour depends on *absence* of a flag rather than presence of one:
//!
//! * **Matching is on.** A search matches unless `--no-match` says otherwise.
//!   `--bench` implies `--no-match` so it measures the kernel rather than the
//!   matcher, but an explicit `--match` overrides that, so `--bench --match`
//!   measures the full path.
//! * **Format falls back to the file extension.** `--format` always wins; the
//!   extension is consulted only when it is absent, and an unrecognised or
//!   missing extension yields CSV rather than an error.

use std::path::PathBuf;
use std::time::Duration;

use clap::{builder::PossibleValuesParser, Arg, ArgAction, Command};

use crate::output;

/// What the program was asked to do.
pub enum Mode {
    /// Search for the given prefixes.
    Search,
    /// Time the kernel, optionally with matching on or off.
    Bench {
        secs: u64,
        reps: usize,
        /// Prefixes to match against, empty when matching is disabled.
        prefixes: Vec<String>,
    },
    /// Time the whole pipeline for a prefix of a given length.
    BenchPrefix {
        len: usize,
        secs: u64,
        reps: usize,
    },
}

/// Fully resolved options.
pub struct Options {
    pub mode: Mode,
    /// Requested prefixes, for a search.
    pub prefixes: Vec<String>,
    /// Where matches go: one file, or a directory of per-key files.
    pub out: Option<PathBuf>,
    pub format: Option<output::Format>,
    pub dst: Option<PathBuf>,
    /// Add to an existing `--out` file instead of replacing it.
    pub append: bool,
    /// Replace an existing `--out` file.
    pub overwrite: bool,
    /// Stop after this many keys, zero for no limit.
    pub count: u64,
    /// Seconds between progress lines; zero silences them.
    pub log_interval: Duration,
    pub verbose: bool,
    pub threads: Option<u32>,
    pub blocks_per_sm: Option<u32>,
    /// How hard to push each key to disk.
    pub durability: output::Durability,
}

/// Parse the process arguments.
pub fn parse() -> Options {
    let m = app().get_matches();

    let benchmarking = m.get_one::<String>("bench").is_some()
        || m.get_one::<String>("bench-prefix").is_some();
    let match_enabled = if m.get_flag("no-match") {
        false
    } else if m.get_flag("match") {
        true
    } else {
        !benchmarking
    };

    // Duration and repetition count, shared by both benchmark modes.
    //
    // `--bench N` and `--bench-prefix N` both take seconds, and `--bench-prefix`
    // falls back to the same default. That was already true; what was *not*
    // explicit was that `--reps` and `--quick` could silently disagree, since
    // `--quick` was only consulted when `--reps` was absent. An explicit
    // `--reps 5 --quick` had no defined meaning and quietly did 5 reps. Now
    // `--quick` is exactly "one rep" and an explicit `--reps` wins, so the
    // combination is at least deterministic.
    let secs: u64 = m
        .get_one::<String>("bench")
        .or_else(|| m.get_one::<String>("bench-prefix"))
        .map(|s| s.parse().expect("benchmark duration takes a number of seconds"))
        .unwrap_or(30);
    let reps: usize = match m.get_one::<String>("reps") {
        Some(r) => r.parse().expect("--reps takes a count"),
        None if m.get_flag("quick") => 1,
        None => 3,
    };
    assert!(reps >= 1, "--reps must be at least 1, got {reps}");
    assert!(secs >= 1, "benchmark duration must be at least 1 second, got {secs}");

    // `--bench-prefix` wins over `--bench`: it is the more specific request, and
    // `--bench` doubles as its duration.
    let mode = match m.get_one::<String>("bench-prefix") {
        Some(len) => {
            let len: usize = len.parse().unwrap_or_else(|_| {
                panic!(
                    "--bench-prefix takes a prefix length in characters, got {:?}; \
                     to set the duration use --bench, as in \
                     --bench 10 --bench-prefix 4",
                    len
                )
            });
            assert!(
                (1..=12).contains(&len),
                "--bench-prefix must be 1..=12 characters, got {}",
                len
            );
            Mode::BenchPrefix { len, secs, reps }
        }
        None if benchmarking => Mode::Bench {
            secs,
            reps,
            // With matching off the kernel only generates keys. With matching
            // on, use the prefix the caller gave, defaulting to a single
            // character so there is something to match against.
            prefixes: if match_enabled {
                vec![m.get_one::<String>("PREFIX").unwrap_or(&"z".to_string()).to_string()]
            } else {
                vec![]
            },
        },
        None => Mode::Search,
    };

    let out = m.get_one::<String>("out").map(PathBuf::from);
    let format = m.get_one::<String>("format").map(|name| {
        output::Format::parse(name).unwrap_or_else(|| {
            panic!(
                "unknown --format {:?}; expected one of {}",
                name,
                output::Format::NAMES.join(", ")
            )
        })
    });
    if format.is_some() && out.is_none() {
        eprintln!("warning: --format has no effect without --out; using per-key files");
    }

    // A Tor v3 address is canonically lowercase, and every consumer of a prefix
    // assumes it: the kernel packs the prefix bits into a base32-decoded byte
    // word, and `main` re-verifies a match with `onion.starts_with(prefix)`.
    // So an uppercase `AB` would decode fine but could never match the
    // lowercase address it names, and the search would run forever finding
    // nothing. Normalise once, here, where the module's own rule puts
    // resolution, and every downstream value is lowercase by construction.
    let prefixes: Vec<String> = m
        .get_many::<String>("PREFIX")
        .map(|v| normalise_prefixes(v.cloned().collect()))
        .unwrap_or_default();
    // `longest` is measured after normalisation, so it is a character count of
    // what will actually be searched rather than of what was typed.
    let longest = prefixes.iter().map(|s| s.len()).max().unwrap_or(0);
    // Durability follows the longest prefix unless overridden. The reasoning is
    // in `output::Durability`: an fsync per key is ruinous at thousands of
    // matches per second and free at a handful, and the long searches are both
    // the ones worth protecting and the ones where it is cheap.
    let durability = if m.get_flag("no-fsync") {
        output::Durability::Buffered
    } else if m.get_flag("fsync") {
        output::Durability::Durable
    } else {
        output::Durability::for_prefix_len(longest)
    };

    // `--quiet` silences the progress line and the configuration summary, so a
    // search can be piped without interleaved chatter. It is applied here rather
    // than at each print site so there is one place that decides.
    let quiet = m.get_flag("quiet");
    let log_interval = match m.get_one::<String>("log-interval") {
        Some(v) => Duration::from_secs(v.parse().expect("--log-interval takes seconds")),
        None if quiet => Duration::from_secs(0),
        None => Duration::from_secs(30),
    };
    let count: u64 = match m.get_one::<String>("count") {
        Some(c) => c.parse().expect("--count takes a number of keys"),
        None => 0,
    };

    Options {
        mode,
        prefixes,
        out,
        format,
        dst: m.get_one::<String>("dst").map(PathBuf::from),
        append: m.get_flag("append"),
        overwrite: m.get_flag("overwrite"),
        count,
        log_interval,
        verbose: m.get_flag("verbose") && !quiet,
        threads: m
            .get_one::<String>("threads")
            .map(|t| t.parse().expect("--threads takes a count")),
        blocks_per_sm: m
            .get_one::<String>("blocks-per-sm")
            .map(|p| p.parse().expect("--blocks-per-sm takes a count")),
        durability,
    }
}

fn app() -> Command {
    Command::new("t3v")
        .version(env!("CARGO_PKG_VERSION"))
        .about("Generate Tor v3 vanity .onion addresses by brute-forcing ed25519 keys on an NVIDIA GPU")
        .arg(
            Arg::new("PREFIX")
                .required_unless_present_any(["bench", "bench-prefix"])
                .action(ArgAction::Append)
                .value_delimiter(',')
                .value_name("PREFIX")
                .help("Desired prefix, base32 (a-z, 2-7). Repeat or comma-separate for \
                       several; case is ignored"),
        )
        // --- where matches go -------------------------------------------------
        .arg(
            Arg::new("dst")
                .long("dst")
                .short('d')
                .num_args(1)
                .value_name("DIR")
                .help("Destination folder for per-key files (default: cwd)"),
        )
        .arg(
            Arg::new("out")
                .long("out")
                .short('o')
                .num_args(1)
                .value_name("FILE")
                .help("Write every match to this single file instead of many small ones"),
        )
        .arg(
            Arg::new("append")
                .long("append")
                .short('a')
                .action(ArgAction::SetTrue)
                .conflicts_with_all(["dst", "overwrite"])
                .requires("out")
                .help("Add to an existing --out file rather than replacing it, which is \
                       what makes a long search resumable. A CSV keeps its one header, \
                       and a .json array has its closing bracket removed so the new \
                       records stay inside it"),
        )
        .arg(
            Arg::new("overwrite")
                .long("overwrite")
                .action(ArgAction::SetTrue)
                .conflicts_with_all(["dst", "append"])
                .requires("out")
                .help("Replace an existing --out file, discarding what is in it. \
                       Without this, or --append, an existing file is an error rather \
                       than something this tool silently deletes"),
        )
        .arg(
            Arg::new("format")
                .long("format")
                .short('f')
                .num_args(1)
                .value_name("FORMAT")
                .value_parser(PossibleValuesParser::new(output::Format::NAMES))
                .help("Encoding for --out. Defaults to the file extension, else csv. \
                       Setting it explicitly lets any filename hold any format"),
        )
        // --- what the kernel does ---------------------------------------------
        //
        // `--match` and `--no-match` are a pair and are declared as one, so clap
        // rejects `--match --no-match` instead of letting precedence decide.
        // Previously both could be given and the result depended on which the
        // parser checked first -- the kind of thing that is only noticed when it
        // produces the wrong answer.
        .arg(
            Arg::new("match")
                .long("match")
                .action(ArgAction::SetTrue)
                .conflicts_with("no-match")
                .help("Check candidates against the prefix (default). The kernel \
                       skips compression and the batch inversion when this is off"),
        )
        .arg(
            Arg::new("no-match")
                .long("no-match")
                .action(ArgAction::SetTrue)
                .conflicts_with("match")
                .help("Only generate keys; never test them against a prefix"),
        )
        // --- benchmarking ------------------------------------------------------
        .arg(
            Arg::new("bench")
                .long("bench")
                .short('b')
                .num_args(1)
                .value_name("SECONDS")
                .help("Benchmark for this many seconds, then exit (default 30). \
                       Implies --no-match; pass --match to include matching. Doubles \
                       as the duration for --bench-prefix"),
        )
        .arg(
            Arg::new("bench-prefix")
                .long("bench-prefix")
                .short('P')
                .num_args(1)
                .value_name("CHARS")
                .help("Benchmark end-to-end for an N-character prefix: the number \
                       that actually matters, keys *found* per second"),
        )
        .arg(
            Arg::new("reps")
                .long("reps")
                .short('r')
                .num_args(1)
                .value_name("N")
                .requires("bench")
                .help("Benchmark repetitions, best is reported (default 3, or 1 \
                       with --quick)"),
        )
        .arg(
            Arg::new("quick")
                .long("quick")
                .short('Q')
                .action(ArgAction::SetTrue)
                .requires("bench")
                .help("With a benchmark, do a single timed run"),
        )
        // --- how long to run ---------------------------------------------------
        .arg(
            Arg::new("count")
                .long("count")
                .short('n')
                .num_args(1)
                .value_name("KEYS")
                .help("Stop after finding this many keys, then exit cleanly. 0, the \
                       default, means run until interrupted. Useful for collecting a \
                       fixed number of keys without watching for them"),
        )
        // --- logging -----------------------------------------------------------
        .arg(
            Arg::new("log-interval")
                .long("log-interval")
                .short('L')
                .num_args(1)
                .value_name("SECONDS")
                .help("Seconds between progress lines during a search (default 30, \
                       0 to silence)"),
        )
        .arg(
            Arg::new("verbose")
                .long("verbose")
                .short('v')
                .action(ArgAction::SetTrue)
                .conflicts_with("quiet")
                .help("Print the resolved configuration: launch geometry, register \
                       usage, matching, and output format"),
        )
        .arg(
            Arg::new("quiet")
                .long("quiet")
                .short('q')
                .action(ArgAction::SetTrue)
                .conflicts_with("verbose")
                .help("Print nothing but found keys. Implies a silent progress line and \
                       no configuration summary, so a search can be piped without \
                       interleaved chatter"),
        )
        // --- launch geometry ---------------------------------------------------
        //
        // `-p` used to be --blocks-per-sm, which read as "prefix" and collided
        // with the positional. These are both overrides for a choice the program
        // already makes well from the kernel's register count, so they are
        // grouped as `overrides` and the short flags follow the noun: -t for
        // threads, -b is taken by --bench so blocks gets -B.
        .arg(
            Arg::new("threads")
                .long("threads")
                .short('t')
                .num_args(1)
                .value_name("N")
                .help("Threads per block. Default: derived from the kernel's register \
                       usage, which is what actually limits occupancy"),
        )
        .arg(
            Arg::new("blocks-per-sm")
                .long("blocks-per-sm")
                .short('B')
                .num_args(1)
                .value_name("N")
                .help("Blocks per SM to launch. Default: fill the device"),
        )
        // --- durability -------------------------------------------------------
        //
        // A pair, like --match/--no-match, and rejected when both are given.
        .arg(
            Arg::new("fsync")
                .long("fsync")
                .action(ArgAction::SetTrue)
                .conflicts_with("no-fsync")
                .help("fsync every key, so it survives a crash or power loss. \
                       The default already follows the prefix length: on for 5 \
                       characters and longer, where a key arrives every few \
                       seconds and durability is free; off below that, where \
                       keys arrive thousands per second and it is not"),
        )
        .arg(
            Arg::new("no-fsync")
                .long("no-fsync")
                .action(ArgAction::SetTrue)
                .conflicts_with("fsync")
                .help("Buffer output even for long prefixes, trading durability \
                       for throughput"),
        )
}

/// Normalise the prefixes the way `parse` does, as a pure function.
///
/// Extracted so the rule can be tested without constructing a command line.
/// The property that matters is not "lowercases" but the combination that
/// `gpu::prefix_spec` depends on: every prefix reaching it is lowercase, which is
/// what makes `Rfc4648Lower` the correct alphabet. Get this wrong and base32
/// returns `None`, the search runs, and it reports nothing -- the worst failure
/// mode in the program, since a user waiting on a 2^40 search has no way to tell
/// "no match yet" from "broken".
fn normalise_prefixes(raw: Vec<String>) -> Vec<String> {
    raw.into_iter()
        .map(|s| s.to_ascii_lowercase())
        .collect()
}

#[cfg(test)]
mod prefix_tests {
    use super::normalise_prefixes;

    #[test]
    fn uppercase_prefixes_are_lowercased() {
        assert_eq!(normalise_prefixes(vec!["AB".into()]), vec!["ab".to_string()]);
        assert_eq!(
            normalise_prefixes(vec!["TorProject".into()]),
            vec!["torproject".to_string()]
        );
    }

    #[test]
    fn every_normalised_prefix_decodes_with_the_lowercase_alphabet() {
        // The load-bearing property. `gpu::prefix_spec` decodes with
        // `Rfc4648Lower`, which returns `None` for uppercase input. So a prefix
        // that is still uppercase when it arrives there yields no bytes, the
        // search finds nothing, and nothing reports why.
        //
        // Checked by decoding, not by string comparison: the failure is not
        // visible in the string, only in what the decoder does with it.
        const ALPHABET: &[u8] = b"abcdefghijklmnopqrstuvwxyz234567";
        for &c in ALPHABET {
            for typed in [c, c.to_ascii_uppercase()] {
                let got = normalise_prefixes(vec![String::from(typed as char)]);
                assert!(base32::decode(
                    base32::Alphabet::Rfc4648Lower { padding: false },
                    &format!("{}aa", got[0])
                )
                .is_some());
            }
        }
    }

    #[test]
    fn normalisation_is_idempotent() {
        let once = normalise_prefixes(vec!["MIXEDcase7".into()]);
        let twice = normalise_prefixes(once.clone());
        assert_eq!(once, twice);
    }

    #[test]
    fn several_prefixes_are_all_normalised() {
        // Multiple prefixes are a documented feature, and the per-key-file
        // output names each one, so a missed normalisation would also produce
        // wrongly-cased filenames.
        assert_eq!(
            normalise_prefixes(vec!["AB".into(), "cd".into(), "EF".into()]),
            vec!["ab".to_string(), "cd".to_string(), "ef".to_string()]
        );
    }
}
