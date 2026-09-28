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

    let benchmarking = m.is_present("bench") || m.is_present("bench-prefix");
    let match_enabled = if m.is_present("no-match") {
        false
    } else if m.is_present("match") {
        true
    } else {
        !benchmarking
    };

    let secs: u64 = m
        .value_of("bench")
        .map(|s| s.parse().expect("--bench takes a number of seconds"))
        .unwrap_or(30);
    let reps: usize = match m.value_of("reps") {
        Some(r) => r.parse().expect("--reps takes a count"),
        None if m.is_present("quick") => 1,
        None => 3,
    };

    // `--bench-prefix` wins over `--bench`: it is the more specific request, and
    // `--bench` doubles as its duration.
    let mode = match m.value_of("bench-prefix") {
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
                vec![m.value_of("PREFIX").unwrap_or("z").to_string()]
            } else {
                vec![]
            },
        },
        None => Mode::Search,
    };

    let out = m.value_of("out").map(PathBuf::from);
    let format = m.value_of("format").map(|name| {
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

    // Durability follows the longest prefix unless overridden. The reasoning is
    // in `output::Durability`: an fsync per key is ruinous at thousands of
    // matches per second and free at a handful, and the long searches are both
    // the ones worth protecting and the ones where it is cheap.
    let longest = m
        .values_of("PREFIX")
        .map(|v| v.map(|s| s.len()).max().unwrap_or(0))
        .unwrap_or(0);
    let durability = if m.is_present("no-fsync") {
        output::Durability::Buffered
    } else if m.is_present("fsync") {
        output::Durability::Durable
    } else {
        output::Durability::for_prefix_len(longest)
    };

    Options {
        mode,
        prefixes: m
            .values_of("PREFIX")
            .map(|v| v.map(|s| s.to_string()).collect())
            .unwrap_or_default(),
        out,
        format,
        dst: m.value_of("dst").map(PathBuf::from),
        log_interval: match m.value_of("log-interval") {
            Some(v) => Duration::from_secs(v.parse().expect("--log-interval takes seconds")),
            None => Duration::from_secs(30),
        },
        verbose: m.is_present("verbose"),
        threads: m
            .value_of("threads")
            .map(|t| t.parse().expect("--threads takes a count")),
        blocks_per_sm: m
            .value_of("blocks-per-sm")
            .map(|p| p.parse().expect("--blocks-per-sm takes a count")),
        durability: durability,
    }
}

fn app() -> clap::App<'static, 'static> {
    clap::App::new("t3v")
        .arg(
            clap::Arg::with_name("PREFIX")
                .required_unless_one(&["bench", "bench-prefix"])
                .multiple(true)
                .value_delimiter(",")
                .help("Desired prefix"),
        )
        // --- where matches go -------------------------------------------------
        .arg(
            clap::Arg::with_name("dst")
                .long("dst")
                .short("d")
                .takes_value(true)
                .help("Destination folder for per-key files (default: cwd)"),
        )
        .arg(
            clap::Arg::with_name("out")
                .long("out")
                .short("o")
                .takes_value(true)
                .help("Write every match to this single file instead of many small ones"),
        )
        .arg(
            clap::Arg::with_name("format")
                .long("format")
                .short("f")
                .takes_value(true)
                .possible_values(output::Format::NAMES)
                .help("Encoding for --out. Defaults to the file extension, else csv. \
                       Setting it explicitly lets any filename hold any format"),
        )
        // --- what the kernel does ---------------------------------------------
        .arg(
            clap::Arg::with_name("match")
                .long("match")
                .help("Check candidates against the prefix (default). The kernel \
                       skips compression and the batch inversion when this is off"),
        )
        .arg(
            clap::Arg::with_name("no-match")
                .long("no-match")
                .help("Only generate keys; never test them against a prefix"),
        )
        // --- benchmarking ------------------------------------------------------
        .arg(
            clap::Arg::with_name("bench")
                .long("bench")
                .short("b")
                .takes_value(true)
                .help("Benchmark for this many seconds, then exit (default 30). \
                       Implies --no-match; pass --match to include matching"),
        )
        .arg(
            clap::Arg::with_name("bench-prefix")
                .long("bench-prefix")
                .takes_value(true)
                .help("Benchmark end-to-end for an N-character prefix: the number \
                       that actually matters, keys *found* per second"),
        )
        .arg(
            clap::Arg::with_name("reps")
                .long("reps")
                .short("r")
                .takes_value(true)
                .help("Benchmark repetitions, best is reported (default 3, or 1 \
                       with --quick)"),
        )
        .arg(
            clap::Arg::with_name("quick")
                .long("quick")
                .short("Q")
                .help("With a benchmark, do a single timed run"),
        )
        // --- logging -----------------------------------------------------------
        .arg(
            clap::Arg::with_name("log-interval")
                .long("log-interval")
                .short("L")
                .takes_value(true)
                .help("Seconds between progress lines during a search (default 30, \
                       0 to silence)"),
        )
        .arg(
            clap::Arg::with_name("verbose")
                .long("verbose")
                .short("v")
                .help("Print the resolved configuration: launch geometry, register \
                       usage, matching, and output format"),
        )
        // --- launch geometry ---------------------------------------------------
        .arg(
            clap::Arg::with_name("threads")
                .long("threads")
                .short("t")
                .takes_value(true)
                .help("Threads per block. Default: derived from the kernel's register \
                       usage, which is what actually limits occupancy"),
        )
        .arg(
            clap::Arg::with_name("blocks-per-sm")
                .long("blocks-per-sm")
                .short("p")
                .takes_value(true)
                .help("Blocks per SM to launch. Default: fill the device"),
        )
        // --- durability -------------------------------------------------------
        .arg(
            clap::Arg::with_name("fsync")
                .long("fsync")
                .help("fsync every key, so it survives a crash or power loss. \
                       The default already follows the prefix length: on for 5 \
                       characters and longer, where a key arrives every few \
                       seconds and durability is free; off below that, where \
                       keys arrive thousands per second and it is not"),
        )
        .arg(
            clap::Arg::with_name("no-fsync")
                .long("no-fsync")
                .help("Buffer output even for long prefixes, trading durability \
                       for throughput"),
        )
}
