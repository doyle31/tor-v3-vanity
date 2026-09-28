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

    let secs: u64 = m
        .get_one::<String>("bench")
        .map(|s| s.parse().expect("--bench takes a number of seconds"))
        .unwrap_or(30);
    let reps: usize = match m.get_one::<String>("reps") {
        Some(r) => r.parse().expect("--reps takes a count"),
        None if m.get_flag("quick") => 1,
        None => 3,
    };

    // `--bench-prefix` wins over `--bench`: it is the more specific request, and
    // `--bench` doubles as its duration.
    let mode = match m.get_one::<String>("bench-prefix") {
        Some(len) => {
            let len: usize = len
                .parse()
                .unwrap_or_else(|_| panic!("--bench-prefix takes a length, got {:?}", len));
            assert!(
                (1..=12).contains(&len),
                "--bench-prefix must be 1..=12, got {}",
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
        .map(|v| v.map(|s| s.to_ascii_lowercase()).collect())
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

    Options {
        mode,
        prefixes,
        out,
        format,
        dst: m.get_one::<String>("dst").map(PathBuf::from),
        log_interval: match m.get_one::<String>("log-interval") {
            Some(v) => Duration::from_secs(v.parse().expect("--log-interval takes seconds")),
            None => Duration::from_secs(30),
        },
        verbose: m.get_flag("verbose"),
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
        .arg(
            Arg::new("PREFIX")
                .required_unless_present_any(["bench", "bench-prefix"])
                .action(ArgAction::Append)
                .value_delimiter(',')
                .help("Desired prefix"),
        )
        // --- where matches go -------------------------------------------------
        .arg(
            Arg::new("dst")
                .long("dst")
                .short('d')
                .num_args(1)
                .help("Destination folder for per-key files (default: cwd)"),
        )
        .arg(
            Arg::new("out")
                .long("out")
                .short('o')
                .num_args(1)
                .help("Write every match to this single file instead of many small ones"),
        )
        .arg(
            Arg::new("format")
                .long("format")
                .short('f')
                .num_args(1)
                .value_parser(PossibleValuesParser::new(output::Format::NAMES))
                .help("Encoding for --out. Defaults to the file extension, else csv. \
                       Setting it explicitly lets any filename hold any format"),
        )
        // --- what the kernel does ---------------------------------------------
        .arg(
            Arg::new("match")
                .long("match")
                .action(ArgAction::SetTrue)
                .help("Check candidates against the prefix (default). The kernel \
                       skips compression and the batch inversion when this is off"),
        )
        .arg(
            Arg::new("no-match")
                .long("no-match")
                .action(ArgAction::SetTrue)
                .help("Only generate keys; never test them against a prefix"),
        )
        // --- benchmarking ------------------------------------------------------
        .arg(
            Arg::new("bench")
                .long("bench")
                .short('b')
                .num_args(1)
                .help("Benchmark for this many seconds, then exit (default 30). \
                       Implies --no-match; pass --match to include matching"),
        )
        .arg(
            Arg::new("bench-prefix")
                .long("bench-prefix")
                .num_args(1)
                .help("Benchmark end-to-end for an N-character prefix: the number \
                       that actually matters, keys *found* per second"),
        )
        .arg(
            Arg::new("reps")
                .long("reps")
                .short('r')
                .num_args(1)
                .help("Benchmark repetitions, best is reported (default 3, or 1 \
                       with --quick)"),
        )
        .arg(
            Arg::new("quick")
                .long("quick")
                .short('Q')
                .action(ArgAction::SetTrue)
                .help("With a benchmark, do a single timed run"),
        )
        // --- logging -----------------------------------------------------------
        .arg(
            Arg::new("log-interval")
                .long("log-interval")
                .short('L')
                .num_args(1)
                .help("Seconds between progress lines during a search (default 30, \
                       0 to silence)"),
        )
        .arg(
            Arg::new("verbose")
                .long("verbose")
                .short('v')
                .action(ArgAction::SetTrue)
                .help("Print the resolved configuration: launch geometry, register \
                       usage, matching, and output format"),
        )
        // --- launch geometry ---------------------------------------------------
        .arg(
            Arg::new("threads")
                .long("threads")
                .short('t')
                .num_args(1)
                .help("Threads per block. Default: derived from the kernel's register \
                       usage, which is what actually limits occupancy"),
        )
        .arg(
            Arg::new("blocks-per-sm")
                .long("blocks-per-sm")
                .short('p')
                .num_args(1)
                .help("Blocks per SM to launch. Default: fill the device"),
        )
        // --- durability -------------------------------------------------------
        .arg(
            Arg::new("fsync")
                .long("fsync")
                .action(ArgAction::SetTrue)
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
                .help("Buffer output even for long prefixes, trading durability \
                       for throughput"),
        )
}
