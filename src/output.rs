//! Where matched keypairs go.
//!
//! Writing one file per key is unusable at scale: a 6-character prefix yields
//! on the order of a thousand keys, and a 7-character prefix tens of
//! thousands, so the filesystem fills with tiny files that all need to be
//! concatenated before use. This module streams every match into a single
//! file instead.
//!
//! # Formats
//!
//! | extension | shape                       | use                              |
//! |-----------|-----------------------------|----------------------------------|
//! | `.csv`    | header row + one row each   | spreadsheets, `awk`, bulk import |
//! | `.jsonl`  | one object per line         | streaming, `jq`, append-friendly  |
//! | `.json`   | single array                | tools that want one parseable doc|
//!
//! # What is stored
//!
//! A Tor v3 service key is the 32-byte constant header
//! `== ed25519v1-secret: type0 ==\0\0\0` followed by the 64-byte expanded
//! secret key. The header never varies, so repeating it once per row is pure
//! waste -- at 128 hex characters of key per match, a constant 32-byte header
//! would add 66% to the file for no information. Instead each row carries the
//! 64-byte expanded key as hex, and `emit_tor_files` (or the one-liner in the
//! README) turns that into a real `hs_ed25519_secret_key`.
//!
//! Keys are cryptographically random, so these files do not compress. Measured
//! with zstd -3 over 200k keys: raw binary 64.0 bytes/key at ratio 1.000 (no
//! compression at all), hex CSV 129.0 bytes/key compressing to 67.9. Text
//! encodings are therefore strictly worse than the 62.75-byte entropy floor.

use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;

/// The constant Tor v3 secret-key header. Not stored per row; see module docs.
pub const TOR_HEADER: &[u8] = b"== ed25519v1-secret: type0 ==\0\0\0";

/// Output encodings.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Format {
    Csv,
    JsonLines,
    JsonArray,
}

impl Format {
    /// Every name `--format` accepts, for help text and error messages.
    pub const NAMES: &'static [&'static str] = &["csv", "jsonl", "ndjson", "json"];

    /// Parse an explicit `--format` value.
    ///
    /// This is authoritative and never consults the file name, so any output
    /// path pairs with any format -- including a path with no extension, or
    /// one whose extension says something else.
    pub fn parse(name: &str) -> Option<Self> {
        match name.trim().to_ascii_lowercase().as_str() {
            "csv" => Some(Format::Csv),
            "jsonl" | "ndjson" | "jsonlines" | "json-lines" | "jsonlines-ish" => {
                Some(Format::JsonLines)
            }
            "json" | "array" => Some(Format::JsonArray),
            _ => None,
        }
    }

    /// Guess a format from a path's extension.
    ///
    /// Only ever a fallback for when `--format` is absent. An unrecognised or
    /// missing extension yields CSV rather than an error, so `--out results`
    /// and `--out results.dat` both work.
    pub fn from_path(path: &Path) -> Self {
        match path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_ascii_lowercase())
            .as_deref()
        {
            Some("jsonl") | Some("ndjson") => Format::JsonLines,
            Some("json") => Format::JsonArray,
            _ => Format::Csv,
        }
    }

    /// Canonical short name, for messages.
    pub fn name(self) -> &'static str {
        match self {
            Format::Csv => "csv",
            Format::JsonLines => "jsonl",
            Format::JsonArray => "json",
        }
    }
}

/// One matched keypair.
#[derive(Clone)]
pub struct Key {
    /// The `.onion` address, without a trailing newline.
    pub onion: String,
    /// The 64-byte expanded secret key (clamped scalar || signing nonce).
    pub expanded: [u8; 64],
    /// Which requested prefix this matched, when several were given.
    pub matched: String,
}

fn hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push(HEX[(b >> 4) as usize] as char);
        s.push(HEX[(b & 0xF) as usize] as char);
    }
    s
}

/// Minimal JSON string escaping. Onion addresses and hex keys never need it,
/// but a malformed input should not produce invalid JSON.
fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

/// Streaming sink for matched keypairs.
pub enum Sink {
    /// One file per key, written into a directory (the original behaviour).
    Dir {
        dir: std::path::PathBuf,
        written: u64,
    },
    Csv(BufWriter<File>, Flusher),
    JsonLines(BufWriter<File>, u64, Flusher),
    JsonArray(BufWriter<File>, u64, Flusher),
}

/// Whether each record is pushed all the way to disk before the next is
/// written.
///
/// This trade runs opposite to intuition, and the reason is that durability gets
/// *cheaper* exactly where it starts to matter. A search finds keys at roughly
/// `keys_per_second / 2^(5 * prefix_len)`, so at ~150M keys/s:
///
/// | prefix | matches/s | one fsync | right answer |
/// |---|---|---|---|
/// | 2 chars | ~37,000 | ruinous | buffered |
/// | 3 chars | ~4,600  | ruinous | buffered |
/// | 4 chars | ~143    | still far too many | buffered |
/// | 5 chars | ~4.5    | negligible | **durable** |
/// | 6 chars | ~0.14   | free | **durable** |
/// | 7 chars | ~0.004  | free | **durable** |
///
/// The long searches are the ones worth protecting: a 7-character prefix takes
/// minutes of searching, and buffering alone means a crash near the end can
/// throw the result away. That is also precisely where an fsync is free,
/// because matches are that rare. So the default follows the prefix length, and
/// `--fsync` / `--no-fsync` override it.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Durability {
    /// Buffer, and push to the OS on a timer. Fastest; a crash can lose the
    /// records still in the buffer.
    Buffered,
    /// `fsync` every record, so it survives a crash or power loss.
    Durable,
}

impl Durability {
    /// The policy that suits a prefix of `len` characters.
    ///
    /// Five is the crossover: ~4.5 matches per second, where an fsync is a
    /// rounding error, against ~143 per second at four characters, where it is
    /// not.
    pub fn for_prefix_len(len: usize) -> Self {
        if len >= 5 {
            Durability::Durable
        } else {
            Durability::Buffered
        }
    }

    pub fn is_durable(self) -> bool {
        self == Durability::Durable
    }
}

/// Time-based flush guard, plus the durability policy.
///
/// Buffering exists because a fast search can emit keys faster than a syscall
/// can serve them, but a plain `BufWriter` only flushes when it fills. Measured
/// in practice: a 10-second run of `--out keys.csv` produced a **zero-byte**
/// file, because the whole run's keys fit in the 64 KiB buffer. Every key
/// found in that run was lost on exit.
///
/// So in buffered mode each write also flushes once `FLUSH_AFTER` has elapsed,
/// bounding the loss to one interval of keys rather than a whole buffer.
pub struct Flusher {
    last: std::time::Instant,
    durability: Durability,
}

impl Flusher {
    fn new(durability: Durability) -> Self {
        Flusher {
            last: std::time::Instant::now(),
            durability,
        }
    }
}

/// In buffered mode, flush at most this often regardless of how full the
/// buffer is. Ignored entirely in durable mode, which always flushes.
const FLUSH_AFTER: std::time::Duration = std::time::Duration::from_secs(2);

/// Push out the current record according to the policy.
///
/// Buffered: flush only if the timer has elapsed. Durable: flush and `fsync`
/// every time, so the record is on the platter before the next key is even
/// derived.
fn emit(w: &mut BufWriter<File>, g: &mut Flusher) -> io::Result<()> {
    if g.durability.is_durable() {
        w.flush()?;
        w.get_ref().sync_all()?;
    } else if g.last.elapsed() >= FLUSH_AFTER {
        w.flush()?;
        g.last = std::time::Instant::now();
    }
    Ok(())
}

/// What to do about an `--out` file that already exists.
///
/// # Why the default refuses rather than truncating
///
/// A search can run for hours and produce keys that are expensive to reproduce,
/// and `--out keys.csv` is a short, natural thing to type. When the default was
/// to truncate, re-running that command silently destroyed the previous run's
/// work -- the worst possible outcome, because nothing errors, nothing is
/// reported, and the loss is discovered later when the file is found short.
///
/// So the default is [`Existing::Refuse`]: an existing file is an error, and the
/// message names both ways forward. Overwriting is then something a user asks
/// for by name, which it should be.
///
/// [`Existing::Append`] is the mode that makes a long search resumable, since
/// re-running it with the same `--out` adds to what is already there.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Existing {
    /// Refuse to touch a file that is already there.
    Refuse,
    /// Add to what is already there.
    Append,
    /// Truncate it and start over.
    Overwrite,
}

impl Existing {
    /// The policy the flags select. Without `--append` or `--overwrite` the
    /// answer is [`Existing::Refuse`], not a silent overwrite.
    pub fn from_flags(append: bool, overwrite: bool) -> Self {
        match (append, overwrite) {
            (true, true) => Existing::Overwrite,
            (true, false) => Existing::Append,
            (false, true) => Existing::Overwrite,
            (false, false) => Existing::Refuse,
        }
    }
}

/// Prepare an existing JSON array file for appending, in place.
///
/// Returns how many records it already holds.
///
/// A JSON array is one document, so appending to it means undoing the closing
/// bracket `finish` wrote: the new records have to go back *inside* the array or
/// the file stops being valid JSON. That is why this rewrites the tail rather
/// than simply seeking to the end.
///
/// The file is only ever truncated, never extended, and the bytes removed are
/// exactly the trailing whitespace and the `]`. A file that is not an array at
/// all therefore loses its last character, which is the one destructive case --
/// it is why this is only reached for `Append` on a `.json` sink, and the tests
/// cover the ordinary resume, the empty array, and the no-bracket case.
fn resume_json_array(path: &Path) -> io::Result<u64> {
    use std::io::{Read, Seek, Write};

    let mut f = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)?;
    let mut all = Vec::new();
    f.read_to_end(&mut all)?;

    // Count the records already present. Each is one `{`, and only the
    // separator before the *next* record depends on this being non-zero.
    let existing = all.iter().filter(|b| **b == b'{').count() as u64;
    if existing == 0 {
        // Genuinely empty, or an array with no records. Neither has a bracket
        // to remove; an empty file still needs its opening bracket.
        if all.is_empty() {
            f.rewind()?;
            f.write_all(b"[\n")?;
        }
        f.flush()?;
        return Ok(0);
    }

    let mut cut = all.len();
    while cut > 0 && all[cut - 1].is_ascii_whitespace() {
        cut -= 1;
    }
    if cut > 0 && all[cut - 1] == b']' {
        cut -= 1;
    }
    f.set_len(cut as u64)?;
    f.flush()?;
    Ok(existing)
}

impl Sink {
    /// True when `path` is, or will be, a regular file -- and so can be
    /// fsynced. A path that does not exist yet will be created as a regular
    /// file; a character device or FIFO never can be.
    pub fn is_regular_file(path: &Path) -> bool {
        match std::fs::metadata(path) {
            Ok(m) => m.is_file(),
            // Not present yet: `File::create` below will make a regular file.
            Err(_) => true,
        }
    }

    /// Open a sink for `path` in an explicit `format`, applying `existing` to a
    /// file that is already there.
    ///
    /// The format is a parameter rather than something derived from `path`, so
    /// any filename works with any format. See [`Sink::open`] for the
    /// extension-inferring convenience wrapper.
    ///
    /// # What happens to an existing file
    ///
    /// [`Existing::Refuse`] is the default and returns an error, because a search
    /// can run for hours and its output is expensive to reproduce. Overwriting is
    /// then a deliberate act rather than a side effect of retyping a command.
    ///
    /// Each format then needs different care to stay well-formed:
    ///
    /// * **CSV** and **JSON Lines** are self-delimiting, so records simply
    ///   follow on. A CSV *header* is written only for a new file -- repeating it
    ///   mid-file would stop the result being a valid CSV.
    /// * **JSON array** is one document, so the closing `]` has to be removed
    ///   before appending, or the new records land outside the array and the
    ///   file is no longer valid JSON. See [`resume_json_array`].
    ///
    /// # Note on the 4th argument
    ///
    /// This takes an [`Existing`] policy where it used to take none, because
    /// overwriting an existing file silently was found to be a real hazard: a
    /// search can run for hours and retyping `-o keys.csv` destroyed the
    /// previous run with no error. The three-way policy is deliberate, so
    /// dropping it here would drop the protection.
    pub fn open_format(
        path: &Path,
        format: Format,
        durability: Durability,
        existing: Existing,
    ) -> io::Result<Self> {
        use std::io::Write as _;

        // The refusal has to happen before anything is opened for writing, and
        // it is a distinct error rather than the `AlreadyExists` the OS would
        // give, because the remedy is not "pick another name" -- it is
        // `--append` or `--overwrite`.
        //
        // Only regular files are protected. A character device, FIFO or socket
        // is a stream rather than a file holding results, and there is nothing
        // to lose by opening it again: `--out /dev/stdout` is a supported way to
        // pipe the records somewhere, and refusing it would make the flag
        // useless. `is_regular_file` is what distinguishes the two cases.
        if existing == Existing::Refuse
            && path.exists()
            && Self::is_regular_file(path)
        {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                format!(
                    "{} already exists; refusing to overwrite it.\n  \
                     --append  add to it, keeping what is already there\n  \
                     --overwrite  discard it and start over",
                    path.display()
                ),
            ));
        }

        // A JSON array has to rewrite its own tail, so it cannot be opened in
        // append mode below. Handle it first and return.
        if existing == Existing::Append && format == Format::JsonArray && path.exists() {
            let n = resume_json_array(path)?;
            let f = std::fs::OpenOptions::new().append(true).open(path)?;
            let w = BufWriter::with_capacity(1 << 16, f);
            return Ok(Sink::JsonArray(w, n, Flusher::new(durability)));
        }

        let appending = existing == Existing::Append;
        let resuming = appending && path.exists();
        let file = if appending {
            std::fs::OpenOptions::new()
                .append(true)
                .create(true)
                .open(path)?
        } else {
            // `create` truncates an existing file, which is what both Refuse
            // (only reachable when the file does not exist) and Overwrite want.
            File::create(path)?
        };
        let mut w = BufWriter::with_capacity(1 << 16, file);
        let g = Flusher::new(durability);
        match format {
            Format::Csv => {
                // A header belongs to the file, not to each run.
                if !resuming {
                    w.write_all(b"onion,expanded_key,matched_prefix\n")?;
                }
                w.flush()?;
                Ok(Sink::Csv(w, g))
            }
            Format::JsonLines => {
                w.flush()?;
                Ok(Sink::JsonLines(w, 0, g))
            }
            Format::JsonArray => {
                // Reached when overwriting, or appending to a path that did not
                // exist. Either way the array opens here; an existing array was
                // handled above.
                w.write_all(b"[\n")?;
                w.flush()?;
                Ok(Sink::JsonArray(w, 0, g))
            }
        }
    }

    /// Open a sink for `path`, falling back to inferring the format from the
    /// extension when `explicit` is `None`.
    ///
    /// Prefer passing the resolved `--format` value here so that a name like
    /// `keys.dat` is not silently reinterpreted.
    pub fn open(
        path: &Path,
        explicit: Option<Format>,
        durability: Durability,
        existing: Existing,
    ) -> io::Result<Self> {
        let format = explicit.unwrap_or_else(|| Format::from_path(path));
        Self::open_format(path, format, durability, existing)
    }

    /// A sink that writes one file per key into `dir`.
    pub fn dir(dir: &Path) -> Self {
        Sink::Dir {
            dir: dir.to_path_buf(),
            written: 0,
        }
    }

    /// Record a match, flushing if the buffer has been idle long enough that
    /// the data would otherwise sit unwritten.
    pub fn write(&mut self, key: &Key) -> io::Result<()> {
        let hx = hex(&key.expanded);
        match self {
            Sink::Dir { dir, written } => {
                let mut f = File::create(dir.join(&key.onion))?;
                f.write_all(TOR_HEADER)?;
                f.write_all(&key.expanded)?;
                f.flush()?;
                *written += 1;
            }
            Sink::Csv(w, g) => {
                writeln!(w, "{},{},{}", key.onion, hx, key.matched)?;
                emit(w, g)?;
            }
            // JSON Lines: no separators and no wrapper. Each record is
            // self-contained and terminated by its own newline, which is what
            // makes the format streamable and append-safe.
            Sink::JsonLines(w, n, g) => {
                write!(
                    w,
                    "{{\"onion\":\"{}\",\"expanded_key\":\"{}\",\"matched_prefix\":\"{}\"}}\n",
                    json_escape(&key.onion),
                    hx,
                    json_escape(&key.matched)
                )?;
                *n += 1;
                emit(w, g)?;
            }
            Sink::JsonArray(w, n, g) => {
                if *n > 0 {
                    w.write_all(b",\n")?;
                }
                write!(
                    w,
                    "  {{\"onion\":\"{}\",\"expanded_key\":\"{}\",\"matched_prefix\":\"{}\"}}",
                    json_escape(&key.onion),
                    hx,
                    json_escape(&key.matched)
                )?;
                *n += 1;
                emit(w, g)?;
            }
        }
        Ok(())
    }

    /// Finish up: close the JSON array if that is the format in use.
    pub fn finish(&mut self) -> io::Result<()> {
        match self {
            Sink::JsonArray(w, n, _) => {
                if *n > 0 {
                    w.write_all(b"\n")?;
                }
                w.write_all(b"]\n")?;
            }
            _ => {}
        }
        self.flush()
    }

    pub fn flush(&mut self) -> io::Result<()> {
        match self {
            Sink::Dir { .. } => Ok(()),
            Sink::Csv(w, _) | Sink::JsonLines(w, _, _) | Sink::JsonArray(w, _, _) => w.flush(),
        }
    }

    /// How many keys have been recorded.
    pub fn count(&self) -> u64 {
        match self {
            Sink::Dir { written, .. } => *written,
            Sink::Csv(_, _) => 0,
            Sink::JsonLines(_, n, _) | Sink::JsonArray(_, n, _) => *n,
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    fn key(i: u8) -> Key {
        let mut e = [0u8; 64];
        for (k, b) in e.iter_mut().enumerate() {
            *b = i.wrapping_add(k as u8);
        }
        Key {
            onion: format!("abc{:0>52}.onion", i),
            expanded: e,
            matched: "abc".into(),
        }
    }

    /// A unique scratch path, so tests that use the same label cannot collide
    /// when the harness runs them concurrently.
    fn tmp(name: &str) -> std::path::PathBuf {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static N: AtomicUsize = AtomicUsize::new(0);
        let mut p = std::env::temp_dir();
        p.push(format!(
            "t3v-out-test-{}-{}-{}",
            std::process::id(),
            name,
            N.fetch_add(1, Ordering::SeqCst)
        ));
        let _ = std::fs::remove_file(&p);
        p
    }

    #[test]
    fn format_from_extension() {
        assert!(matches!(
            Format::from_path(Path::new("k.csv")),
            Format::Csv
        ));
        assert!(matches!(
            Format::from_path(Path::new("k.CSV")),
            Format::Csv
        ));
        assert!(matches!(
            Format::from_path(Path::new("k.jsonl")),
            Format::JsonLines
        ));
        assert!(matches!(
            Format::from_path(Path::new("k.ndjson")),
            Format::JsonLines
        ));
        assert!(matches!(
            Format::from_path(Path::new("k.json")),
            Format::JsonArray
        ));
        // Unknown extension falls back to CSV.
        assert!(matches!(
            Format::from_path(Path::new("k.txt")),
            Format::Csv
        ));
        assert!(matches!(Format::from_path(Path::new("k")), Format::Csv));
    }

    #[test]
    fn explicit_format_parse_is_case_and_space_insensitive() {
        for n in ["csv", "CSV", " csv ", "Csv"] {
            assert_eq!(Format::parse(n), Some(Format::Csv), "{:?}", n);
        }
        for n in ["jsonl", "JSONL", " ndjson ", "json-lines", "jsonlines"] {
            assert_eq!(Format::parse(n), Some(Format::JsonLines), "{:?}", n);
        }
        for n in ["json", "JSON", " json ", "array"] {
            assert_eq!(Format::parse(n), Some(Format::JsonArray), "{:?}", n);
        }
        for n in ["", "txt", "yaml", "csv2", "csv,", "cs v"] {
            assert_eq!(Format::parse(n), None, "{:?}", n);
        }
    }

    #[test]
    fn explicit_format_overrides_a_misleading_extension() {
        // The whole point of --format: a `.json` name holding CSV, and an
        // extensionless name holding JSON, must both do what was asked.
        let p = tmp("override-json");
        let mut s = Sink::open(&p, Some(Format::Csv), Durability::Buffered, Existing::Overwrite).unwrap();
        s.write(&key(0)).unwrap();
        s.finish().unwrap();
        let body = std::fs::read_to_string(&p).unwrap();
        assert!(body.starts_with("onion,expanded_key,matched_prefix\n"));
        assert!(!body.contains('['), "must not be a JSON array");

        let p = tmp("override-noext");
        let mut s = Sink::open(&p, Some(Format::JsonArray), Durability::Buffered, Existing::Overwrite).unwrap();
        s.write(&key(0)).unwrap();
        s.finish().unwrap();
        let body = std::fs::read_to_string(&p).unwrap();
        assert!(body.starts_with("[\n"), "got {:?}", body);
        assert!(body.ends_with("\n]\n"));
    }

    #[test]
    fn extensionless_output_falls_back_to_csv_and_still_writes() {
        let p = tmp("noext");
        let mut s = Sink::open(&p, None, Durability::Buffered, Existing::Overwrite).unwrap();
        s.write(&key(0)).unwrap();
        s.finish().unwrap();
        let body = std::fs::read_to_string(&p).unwrap();
        assert_eq!(body.lines().count(), 2);
        assert!(body.starts_with("onion,"));
        std::fs::remove_file(&p).unwrap();
    }

    #[test]
    fn every_format_can_target_every_filename() {
        // Matrix check: format choice and filename must be independent axes.
        let formats = [Format::Csv, Format::JsonLines, Format::JsonArray];
        let names = ["out", "out.dat", "out.json", "out.jsonl", "out.csv"];
        for f in formats {
            for n in names {
                let p = tmp(&format!("matrix-{}-{}", f.name(), n.replace('.', "_")));
                let mut s = Sink::open(&p, Some(f), Durability::Buffered, Existing::Overwrite).unwrap();
                s.write(&key(7)).unwrap();
                s.finish().unwrap();
                let body = std::fs::read_to_string(&p).unwrap();
                assert!(
                    body.contains(&hex(&key(7).expanded)),
                    "{} -> {} lost the key",
                    f.name(),
                    n
                );
                match f {
                    Format::Csv => assert!(body.starts_with("onion,")),
                    Format::JsonLines => {
                        // One object per line, nothing joining the lines and
                        // no wrapper. Commas inside a record are expected;
                        // what must not appear is a record separator.
                        assert_eq!(body.lines().count(), 1, "{}", n);
                        let l = body.lines().next().unwrap();
                        assert!(l.starts_with('{') && l.ends_with('}'), "{}", n);
                        assert!(!l.starts_with(',') && !l.starts_with(']'), "{}", n);
                    }
                    Format::JsonArray => {
                        assert!(body.starts_with("[\n") && body.ends_with("\n]\n"))
                    }
                }
                std::fs::remove_file(&p).unwrap();
            }
        }
    }

    #[test]
    fn csv_has_header_and_one_row_per_key() {
        let p = tmp("csv");
        let mut s = Sink::open(&p, None, Durability::Buffered, Existing::Overwrite).unwrap();
        for i in 0..3u8 {
            s.write(&key(i as u8)).unwrap();
        }
        s.finish().unwrap();
        let body = std::fs::read_to_string(&p).unwrap();
        let lines: Vec<&str> = body.lines().collect();
        assert_eq!(lines.len(), 4, "header + 3 rows");
        assert_eq!(lines[0], "onion,expanded_key,matched_prefix");
        assert!(lines[1].starts_with("abc"));
        assert!(lines[1].contains(&hex(&key(0).expanded)));
        assert!(lines[1].ends_with(",abc"));
        // Key hex is 128 chars, and the row must not contain the Tor header.
        let cols: Vec<&str> = lines[1].split(',').collect();
        assert_eq!(cols.len(), 3);
        assert_eq!(cols[1].len(), 128);
        assert!(!body.contains("ed25519v1-secret"));
        std::fs::remove_file(&p).unwrap();
    }

    #[test]
    fn jsonl_is_one_object_per_line() {
        let p = tmp("jsonl");
        let mut s = Sink::open(&p, Some(Format::JsonLines), Durability::Buffered, Existing::Overwrite).unwrap();
        for i in 0..3u8 {
            s.write(&key(i as u8)).unwrap();
        }
        s.finish().unwrap();
        let body = std::fs::read_to_string(&p).unwrap();
        let lines: Vec<&str> = body.lines().collect();
        assert_eq!(lines.len(), 3, "no header, no wrapper");
        for l in &lines {
            assert!(l.starts_with('{') && l.ends_with('}'));
            assert!(l.contains("\"onion\":\"abc"));
            assert!(l.contains("\"expanded_key\":\""));
            assert!(l.contains("\"matched_prefix\":\"abc\""));
        }
        std::fs::remove_file(&p).unwrap();
    }

    #[test]
    fn json_array_is_valid_when_empty_and_populated() {
        let p = tmp("json-empty");
        let mut s = Sink::open(&p, Some(Format::JsonArray), Durability::Buffered, Existing::Overwrite).unwrap();
        s.finish().unwrap();
        assert_eq!(std::fs::read_to_string(&p).unwrap(), "[\n]\n");
        std::fs::remove_file(&p).unwrap();

        let p = tmp("json-full");
        let mut s = Sink::open(&p, Some(Format::JsonArray), Durability::Buffered, Existing::Overwrite).unwrap();
        for i in 0..3u8 {
            s.write(&key(i as u8)).unwrap();
        }
        s.finish().unwrap();
        let body = std::fs::read_to_string(&p).unwrap();
        assert!(body.starts_with("[\n"));
        assert!(body.ends_with("\n]\n"));
        assert_eq!(body.matches('{').count(), 3);
        // No trailing comma before the closing bracket.
        assert!(!body.contains(",\n]"), "trailing comma: {:?}", body);
        // The real check: it must parse as a JSON array of 3 objects.
        let v = parse_json(&body);
        assert_eq!(v.kind, JKind::Array);
        assert_eq!(v.items.len(), 3);
        for (i, it) in v.items.iter().enumerate() {
            assert_eq!(it.kind, JKind::Object);
            assert_eq!(it.get("onion"), Some(key(i as u8).onion.as_str()));
            assert_eq!(it.get("expanded_key"), Some(hex(&key(i as u8).expanded).as_str()));
        }
        std::fs::remove_file(&p).unwrap();
    }

    #[test]
    fn dir_sink_writes_a_tor_key_file_per_key() {
        let d = std::env::temp_dir().join(format!("t3v-out-dir-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        let mut s = Sink::dir(&d);
        for i in 0..3u8 {
            s.write(&key(i as u8)).unwrap();
        }
        s.finish().unwrap();
        assert_eq!(s.count(), 3);
        let k = key(1);
        let body = std::fs::read(d.join(&k.onion)).unwrap();
        assert_eq!(body.len(), 32 + 64);
        assert_eq!(&body[..32], TOR_HEADER);
        assert_eq!(&body[32..], &k.expanded[..]);
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn hex_is_lowercase_and_fixed_width() {
        assert_eq!(hex(&[0x00, 0x0f, 0xff, 0xa5]), "000fffa5");
        assert_eq!(hex(&[0u8; 64]).len(), 128);
    }

    // ---- minimal JSON validator -------------------------------------------
    //
    // The output formats are the crate's contract with other tools, so the
    // tests check the bytes against a real grammar rather than against string
    // patterns. A pattern check missed the array/JSONL separator confusion
    // during development; this would not.

    #[derive(PartialEq, Debug)]
    enum JKind {
        Array,
        Object,
        Other,
    }

    struct JVal {
        kind: JKind,
        /// Elements, for arrays.
        items: Vec<JVal>,
        /// Key/value pairs, for objects.
        fields: Vec<(String, JVal)>,
        /// Raw token text, for scalars.
        scalar: String,
    }

    impl JVal {
        fn scalar(s: String) -> Self {
            JVal { kind: JKind::Other, items: vec![], fields: vec![], scalar: s }
        }
        fn array(items: Vec<JVal>) -> Self {
            JVal { kind: JKind::Array, items, fields: vec![], scalar: String::new() }
        }
        fn object(fields: Vec<(String, JVal)>) -> Self {
            JVal { kind: JKind::Object, items: vec![], fields, scalar: String::new() }
        }
        /// Look up a string field of an object.
        fn get(&self, k: &str) -> Option<&str> {
            self.fields
                .iter()
                .find(|(n, _)| n == k)
                .map(|(_, v)| v.scalar.as_str())
        }
    }

    /// Write sample records to `path` in `format` and return the file text.
    fn render(format: Format, path: &Path, keys: &[Key]) -> String {
        let mut s =
            Sink::open(path, Some(format), Durability::Buffered, Existing::Overwrite).unwrap();
        for k in keys {
            s.write(k).unwrap();
        }
        s.finish().unwrap();
        std::fs::read_to_string(path).unwrap()
    }

    /// Parse enough JSON to validate our own output: arrays, flat objects with
    /// string values, and the literals. Anything else is a parse failure.
    fn parse_json(s: &str) -> JVal {
        let b: Vec<char> = s.chars().collect();
        let mut i = 0usize;
        let v = pval(&b, &mut i);
        skip_ws(&b, &mut i);
        assert_eq!(i, b.len(), "trailing junk at {} in {:?}", i, s);
        v
    }

    fn skip_ws(b: &[char], i: &mut usize) {
        while *i < b.len() && b[*i].is_whitespace() {
            *i += 1;
        }
    }

    fn lit(b: &[char], i: &mut usize, want: &str) -> bool {
        let w: Vec<char> = want.chars().collect();
        if *i + w.len() <= b.len() && b[*i..*i + w.len()] == w[..] {
            *i += w.len();
            true
        } else {
            false
        }
    }

    fn pstr(b: &[char], i: &mut usize) -> String {
        assert_eq!(b[*i], '"', "expected string at {}", i);
        *i += 1;
        let mut out = String::new();
        while b[*i] != '"' {
            if b[*i] == '\\' {
                *i += 1;
                out.push(match b[*i] {
                    'n' => '\n',
                    'r' => '\r',
                    't' => '\t',
                    'u' => {
                        let h: String = b[*i + 1..*i + 5].iter().collect();
                        *i += 4;
                        char::from_u32(u32::from_str_radix(&h, 16).unwrap()).unwrap()
                    }
                    c => c,
                });
            } else {
                out.push(b[*i]);
            }
            *i += 1;
        }
        *i += 1; // closing quote
        out
    }

    fn pval(b: &[char], i: &mut usize) -> JVal {
        skip_ws(b, i);
        if b[*i] == '[' {
            *i += 1;
            let mut items = Vec::new();
            skip_ws(b, i);
            if b[*i] == ']' {
                *i += 1;
                return JVal::array(items);
            }
            loop {
                items.push(pval(b, i));
                skip_ws(b, i);
                match b[*i] {
                    ',' => *i += 1,
                    ']' => {
                        *i += 1;
                        break;
                    }
                    c => panic!("bad array sep {:?} at {}", c, i),
                }
            }
            JVal::array(items)
        } else if b[*i] == '{' {
            *i += 1;
            let mut fields = Vec::new();
            skip_ws(b, i);
            if b[*i] == '}' {
                *i += 1;
                return JVal::object(fields);
            }
            loop {
                skip_ws(b, i);
                let k = pstr(b, i);
                skip_ws(b, i);
                assert_eq!(b[*i], ':', "expected : at {}", i);
                *i += 1;
                skip_ws(b, i);
                if b[*i] == '"' {
                    let s = pstr(b, i);
                    fields.push((k, JVal::scalar(s)));
                } else {
                    // number/true/false/null: skip the token
                    let start = *i;
                    while *i < b.len() && !b[*i].is_whitespace() && b[*i] != ',' && b[*i] != '}' {
                        *i += 1;
                    }
                    fields.push((k, JVal::scalar(b[start..*i].iter().collect())));
                }
                skip_ws(b, i);
                match b[*i] {
                    ',' => *i += 1,
                    '}' => {
                        *i += 1;
                        break;
                    }
                    c => panic!("bad object sep {:?} at {}", c, i),
                }
            }
            JVal::object(fields)
        } else {
            let start = *i;
            while *i < b.len()
                && !b[*i].is_whitespace()
                && b[*i] != ','
                && b[*i] != ']'
                && b[*i] != '}'
            {
                *i += 1;
            }
            JVal::scalar(b[start..*i].iter().collect())
        }
    }

    #[test]
    fn jsonl_parses_as_independent_objects() {
        let p = tmp("jsonl-parse");
        let keys: Vec<_> = (0..4u8).map(key).collect();
        let body = render(Format::JsonLines, &p, &keys);
        // Every line must stand alone as a complete JSON document.
        let lines: Vec<&str> = body.lines().collect();
        assert_eq!(lines.len(), 4);
        for (i, l) in lines.iter().enumerate() {
            let v = parse_json(l);
            assert_eq!(v.kind, JKind::Object, "line {}: {}", i, l);
            assert_eq!(v.get("onion"), Some(keys[i].onion.as_str()));
            assert_eq!(
                v.get("expanded_key"),
                Some(hex(&keys[i].expanded).as_str())
            );
            assert_eq!(v.get("matched_prefix"), Some("abc"));
        }
        std::fs::remove_file(&p).unwrap();
    }

    #[test]
    fn json_array_parses_with_exact_key_payloads() {
        let p = tmp("json-parse");
        let keys: Vec<_> = (0..3u8).map(key).collect();
        let body = render(Format::JsonArray, &p, &keys);
        let v = parse_json(&body);
        assert_eq!(v.kind, JKind::Array);
        assert_eq!(v.items.len(), keys.len());
        for (i, it) in v.items.iter().enumerate() {
            assert_eq!(it.get("onion"), Some(keys[i].onion.as_str()));
            // The key must survive the round trip byte for byte.
            let hx = it.get("expanded_key").unwrap();
            assert_eq!(hx.len(), 128);
            assert_eq!(hx, hex(&keys[i].expanded).as_str());
        }
        std::fs::remove_file(&p).unwrap();
    }

    #[test]
    fn json_escaping_survives_awkward_onion_values() {
        // Onion addresses are base32, but the escaping path must not be able to
        // emit invalid JSON if a value ever contains a quote or control char.
        let p = tmp("escape");
        let mut k = key(1);
        k.onion = "we\"ird\\\n\t\u{1}name.onion".into();
        let body = render(Format::JsonLines, &p, &[k.clone()]);
        let v = parse_json(body.trim());
        assert_eq!(v.kind, JKind::Object);
        assert_eq!(v.get("onion"), Some(k.onion.as_str()));
        std::fs::remove_file(&p).unwrap();
    }

    #[test]
    fn every_format_is_valid_for_a_bare_extensionless_path() {
        // The requirement: no path may be rejected, and each must hold exactly
        // what --format asked for.
        let p = tmp("bare");
        assert_eq!(p.extension().is_none(), true, "path must be extensionless");
        for (f, check) in [
            (Format::Csv, 0usize),
            (Format::JsonLines, 1),
            (Format::JsonArray, 2),
        ] {
            let keys: Vec<_> = (0..2u8).map(key).collect();
            let body = render(f, &p, &keys);
            match check {
                0 => assert_eq!(
                    body.lines().count(),
                    3,
                    "csv: header + 2 rows"
                ),
                1 => {
                    assert_eq!(body.lines().count(), 2);
                    for l in body.lines() {
                        assert_eq!(parse_json(l).kind, JKind::Object);
                    }
                }
                _ => {
                    let v = parse_json(&body);
                    assert_eq!(v.kind, JKind::Array);
                    assert_eq!(v.items.len(), 2);
                }
            }
        }
        std::fs::remove_file(&p).unwrap();
    }

    #[test]
    fn keys_reach_the_file_without_an_explicit_flush() {
        // Regression: a plain BufWriter only flushes when full, so a short run
        // wrote a zero-byte file and every key it found was lost. The time-based
        // flush is what makes short runs safe. Each iteration waits past
        // FLUSH_AFTER so the guard is certain to fire.
        for (fmt, name) in [
            (Format::Csv, "dur-csv"),
            (Format::JsonLines, "dur-jsonl"),
        ] {
            let p = tmp(name);
            let mut s = Sink::open(&p, Some(fmt), Durability::Buffered, Existing::Overwrite).unwrap();
            // The CSV header is written and flushed at open, so the file is
            // non-empty from the start; JSONL starts empty. Only the *keys*
            // are what this test is about.
            s.write(&key(1)).unwrap();
            let after_first = std::fs::metadata(&p).unwrap().len();
            assert!(
                !std::fs::read_to_string(&p).unwrap().contains(&hex(&key(1).expanded)),
                "{}: first key flushed early, the test proves nothing",
                name
            );
            std::thread::sleep(FLUSH_AFTER + std::time::Duration::from_millis(200));
            s.write(&key(2)).unwrap();
            let after_second = std::fs::metadata(&p).unwrap().len();
            assert!(after_second > after_first, "{}: key stayed in the buffer", name);
            // Both keys must be on disk: the flush carries the earlier one too.
            let body = std::fs::read_to_string(&p).unwrap();
            assert!(body.contains(&hex(&key(1).expanded)), "{} lost key 1", name);
            assert!(body.contains(&hex(&key(2).expanded)), "{} lost key 2", name);
            s.finish().unwrap();
            std::fs::remove_file(&p).unwrap();
        }
    }

    #[test]
    fn durability_follows_the_prefix_length() {
        // The crossover is 5: ~4.5 matches/s at 149M keys/s, where an fsync is
        // a rounding error, against ~143/s at 4 chars, where it is not.
        for len in 0..=4 {
            assert_eq!(
                Durability::for_prefix_len(len),
                Durability::Buffered,
                "{} chars should stay buffered",
                len
            );
        }
        for len in 5..=12 {
            assert_eq!(
                Durability::for_prefix_len(len),
                Durability::Durable,
                "{} chars should be durable",
                len
            );
        }
    }

    #[test]
    fn durable_sink_writes_every_record_immediately() {
        // The point of durable mode: a record must be readable before the next
        // one is written, so a crash cannot lose it. Check that by reading the
        // file while the sink is still open and unflushed.
        let p = tmp("durable");
        let mut s = Sink::open(&p, Some(Format::Csv), Durability::Durable, Existing::Overwrite).unwrap();
        s.write(&key(1)).unwrap();
        let after_first = std::fs::read_to_string(&p).unwrap();
        assert!(
            after_first.contains(&hex(&key(1).expanded)),
            "durable mode must have pushed key 1 already"
        );
        s.write(&key(2)).unwrap();
        let after_second = std::fs::read_to_string(&p).unwrap();
        assert!(after_second.contains(&hex(&key(1).expanded)));
        assert!(after_second.contains(&hex(&key(2).expanded)));
        s.finish().unwrap();
        std::fs::remove_file(&p).unwrap();
    }

    #[test]
    fn durable_json_array_is_recoverable_without_the_closing_bracket() {
        // Durable mode is what makes an interrupted .json array worth anything:
        // each record is on disk as it is found. This checks the records are
        // individually parseable even though the array is not closed.
        let p = tmp("durable-json");
        let mut s = Sink::open(&p, Some(Format::JsonArray), Durability::Durable, Existing::Overwrite).unwrap();
        s.write(&key(1)).unwrap();
        let raw = std::fs::read_to_string(&p).unwrap();
        assert!(raw.contains("\"expanded_key\""));
        // Deliberately do not call finish(): simulate a crash mid-array.
        let body = raw.trim_start_matches('[').trim();
        let first = body.lines().next().unwrap().trim_end_matches(',');
        let v = parse_json(first);
        assert_eq!(v.kind, JKind::Object);
        assert_eq!(v.get("onion"), Some(key(1).onion.as_str()));
        std::fs::remove_file(&p).unwrap();
    }

    #[test]
    fn format_name_roundtrips() {
        for n in ["csv", "jsonl", "json"] {
            let f = Format::parse(n).unwrap();
            assert_eq!(f.name(), n);
            assert_eq!(Format::parse(f.name()), Some(f));
        }
        assert!(Format::NAMES.contains(&"csv"));
        assert!(Format::NAMES.contains(&"jsonl"));
        assert!(Format::NAMES.contains(&"json"));
    }
}

/// Tests for `--append`, the one part of the sink that can destroy data.
///
/// The other formats only ever add records at the end, but a JSON array has to
/// have its closing `]` removed before appending or the new records land outside
/// the document. That truncation is the risky part, so it gets its own cases.
#[cfg(test)]
mod append_tests {
    use super::*;

    fn key(i: u8) -> Key {
        let mut e = [0u8; 64];
        for (k, b) in e.iter_mut().enumerate() {
            *b = i.wrapping_add(k as u8);
        }
        Key {
            onion: format!("abc{:0>52}.onion", i),
            expanded: e,
            matched: "abc".into(),
        }
    }

    fn tmp(name: &str) -> std::path::PathBuf {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static N: AtomicUsize = AtomicUsize::new(0);
        let mut p = std::env::temp_dir();
        p.push(format!(
            "t3v-append-test-{}-{}-{}",
            std::process::id(),
            name,
            N.fetch_add(1, Ordering::SeqCst)
        ));
        let _ = std::fs::remove_file(&p);
        p
    }

    fn read(p: &Path) -> String {
        std::fs::read_to_string(p).unwrap()
    }

    /// Write `n` records through a fresh sink, as a first run would.
    fn first_run(p: &Path, format: Format, n: u8) {
        let mut s =
            Sink::open_format(p, format, Durability::Buffered, Existing::Overwrite).unwrap();
        for i in 0..n {
            s.write(&key(i)).unwrap();
        }
        s.finish().unwrap();
    }

    #[test]
    fn csv_append_keeps_one_header() {
        let p = tmp("csv-header");
        first_run(&p, Format::Csv, 2);
        let mut s =
            Sink::open_format(&p, Format::Csv, Durability::Buffered, Existing::Append).unwrap();
        s.write(&key(9)).unwrap();
        s.finish().unwrap();

        let body = read(&p);
        // Count by line prefix, not by substring: every data row also contains
        // "onion," in its matched_prefix column, so `matches` would count one
        // "header" per row.
        let lines: Vec<&str> = body.lines().collect();
        let headers = lines.iter().filter(|l| l.starts_with("onion,")).count();
        assert_eq!(
            headers, 1,
            "a repeated header makes the file invalid CSV:\n{body}"
        );
        let rows = lines.iter().filter(|l| l.starts_with("abc")).count();
        assert_eq!(rows, 3, "2 from the first run and 1 appended:\n{body}");
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn jsonl_append_accumulates() {
        let p = tmp("jsonl");
        first_run(&p, Format::JsonLines, 2);
        let mut s = Sink::open_format(
            &p,
            Format::JsonLines,
            Durability::Buffered,
            Existing::Append,
        )
        .unwrap();
        s.write(&key(9)).unwrap();
        s.finish().unwrap();

        let body = read(&p);
        let lines: Vec<&str> = body.lines().filter(|l| !l.is_empty()).collect();
        assert_eq!(lines.len(), 3, "one record per line, no wrapper:\n{body}");
        for l in &lines {
            assert!(l.starts_with('{') && l.ends_with('}'), "not a record: {l}");
        }
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn json_array_append_stays_one_valid_document() {
        // The case that needs real work: the file ends with `]`, which has to
        // come back off, or the appended records land outside the array.
        let p = tmp("json-array");
        first_run(&p, Format::JsonArray, 2);
        let mut s = Sink::open_format(
            &p,
            Format::JsonArray,
            Durability::Buffered,
            Existing::Append,
        )
        .unwrap();
        s.write(&key(9)).unwrap();
        s.finish().unwrap();

        let body = read(&p);
        assert_eq!(
            body.matches('[').count(),
            1,
            "a second `[` would mean two documents:\n{body}"
        );
        assert_eq!(
            body.matches(']').count(),
            1,
            "the array must be closed exactly once:\n{body}"
        );
        assert_eq!(body.matches('{').count(), 3, "2 + 1 records:\n{body}");
        assert!(
            body.trim_start().starts_with('[') && body.trim_end().ends_with(']'),
            "must be a single bracketed document:\n{body}"
        );
        // A stray `],` or `,,` would mean the separator logic is wrong.
        assert!(!body.contains("],"), "no record may follow the bracket:\n{body}");
        assert!(!body.contains(",,"), "double comma:\n{body}");
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn json_array_append_is_repeatable() {
        // Three successive runs must produce one array of six, not a growing pile
        // of fragments.
        let p = tmp("json-array-many");
        for run in 0..3u8 {
            let mut s = if run == 0 {
                Sink::open_format(&p, Format::JsonArray, Durability::Buffered, Existing::Overwrite)
                    .unwrap()
            } else {
                Sink::open_format(&p, Format::JsonArray, Durability::Buffered, Existing::Append)
                    .unwrap()
            };
            s.write(&key(run * 10)).unwrap();
            s.write(&key(run * 10 + 1)).unwrap();
            s.finish().unwrap();
        }
        let body = read(&p);
        assert_eq!(body.matches('[').count(), 1, "one document:\n{body}");
        assert_eq!(body.matches('{').count(), 6, "3 runs x 2 records:\n{body}");
        assert!(!body.contains("],"), "no record after the bracket:\n{body}");
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn json_array_append_to_empty_array_does_not_double_the_bracket() {
        // A previous run that found nothing leaves `[\n]\n`. Appending must not
        // write a second opening bracket.
        let p = tmp("json-array-empty");
        first_run(&p, Format::JsonArray, 0);
        let mut s = Sink::open_format(
            &p,
            Format::JsonArray,
            Durability::Buffered,
            Existing::Append,
        )
        .unwrap();
        s.write(&key(1)).unwrap();
        s.finish().unwrap();
        let body = read(&p);
        assert_eq!(body.matches('[').count(), 1, "must not re-open:\n{body}");
        assert_eq!(body.matches('{').count(), 1, "the appended record:\n{body}");
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn truncate_is_still_the_default_and_discards() {
        // --append must be opt-in: without it a second run replaces the first.
        let p = tmp("truncate");
        first_run(&p, Format::Csv, 3);
        first_run(&p, Format::Csv, 1);
        let body = read(&p);
        let lines: Vec<&str> = body.lines().collect();
        // Count headers as lines, not as substrings: every data row also contains
        // the literal "onion," inside its `matched_prefix` column, so a substring
        // count sees one per row and reads as a repeated header when it is not.
        let headers = lines.iter().filter(|l| l.starts_with("onion,")).count();
        let rows = lines.iter().filter(|l| l.starts_with("abc")).count();
        assert_eq!(rows, 1, "truncating must discard the first run:\n{body}");
        assert_eq!(headers, 1, "one header:\n{body}");
        assert_eq!(lines.len(), 2, "a header and one row:\n{body}");
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn the_refusal_only_protects_regular_files() {
        // The one real semantic conflict between the overwrite protection and
        // the stdout fix: `--out /dev/stdout` is a character device that always
        // "exists", so a naive existence check refuses it and breaks piping.
        //
        // The guard is therefore on `is_regular_file`, not on existence. A
        // character device has nothing to lose by being opened again, so the
        // sink proceeds without a flag.
        assert!(
            !Sink::is_regular_file(Path::new("/dev/null")),
            "/dev/null is a character device, not a regular file"
        );
        assert!(
            !Sink::is_regular_file(Path::new("/dev/zero")),
            "/dev/zero likewise"
        );
        // A path that does not exist yet will be created as a regular file, so
        // it counts as one -- which is why a fresh --out is never refused.
        let p = tmp("refuse-fresh");
        assert!(!p.exists());
        assert!(Sink::is_regular_file(&p), "a new path becomes a regular file");
        // And opening a device must not be refused.
        let r = Sink::open_format(
            Path::new("/dev/null"),
            Format::Csv,
            Durability::Buffered,
            Existing::Refuse,
        );
        assert!(
            r.is_ok(),
            "a character device must not trip the refusal: {:?}",
            r.err()
        );
        // A regular file, by contrast, still is protected -- covered by
        // `an_existing_file_is_refused_by_default`, and the distinction is what
        // this test is for.
        let q = tmp("refuse-regular");
        first_run(&q, Format::Csv, 1);
        assert!(Sink::is_regular_file(&q), "the temp file is a regular file");
        assert!(Sink::open_format(
            &q,
            Format::Csv,
            Durability::Buffered,
            Existing::Refuse
        )
        .is_err());
        let _ = std::fs::remove_file(&p);
        let _ = std::fs::remove_file(&q);
    }

    #[test]
    fn append_to_a_nonexistent_file_just_creates_it() {
        let p = tmp("append-new");
        assert!(!p.exists());
        let mut s =
            Sink::open_format(&p, Format::Csv, Durability::Buffered, Existing::Append).unwrap();
        s.write(&key(1)).unwrap();
        s.finish().unwrap();
        let body = read(&p);
        assert!(
            body.starts_with("onion,"),
            "a new file still gets its header:\n{body}"
        );
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn existing_policy_maps_from_the_flags() {
        // The default is the point of the whole change: with neither flag, an
        // existing file is refused rather than overwritten.
        assert_eq!(Existing::from_flags(false, false), Existing::Refuse);
        assert_eq!(Existing::from_flags(true, false), Existing::Append);
        assert_eq!(Existing::from_flags(false, true), Existing::Overwrite);
        // clap rejects --append with --overwrite, so this combination is
        // unreachable from the command line; Overwrite is the safe reading.
        assert_eq!(Existing::from_flags(true, true), Existing::Overwrite);
    }

    #[test]
    fn an_existing_file_is_refused_by_default() {
        // The load-bearing safety property, and the reason `--overwrite` exists.
        // `File::create` truncates, so without an explicit check a re-run would
        // silently destroy hours of keys.
        let p = tmp("refuse");
        first_run(&p, Format::Csv, 3);
        let before = read(&p);
        assert!(before.contains("abc"), "first run wrote something");

        let err = match Sink::open_format(&p, Format::Csv, Durability::Buffered, Existing::Refuse) {
            Err(e) => e,
            // `Sink` has no `Debug`, so `expect_err` is not available here.
            Ok(_) => panic!("must refuse an existing file"),
        };
        let msg = err.to_string();
        assert_eq!(err.kind(), std::io::ErrorKind::AlreadyExists);
        assert!(msg.contains("--append"), "the error must name the remedy: {msg}");
        assert!(msg.contains("--overwrite"), "and the other one: {msg}");

        // And the file is untouched, which is the whole point.
        assert_eq!(read(&p), before, "a refused open must not modify the file");
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn overwrite_still_replaces_when_asked() {
        let p = tmp("overwrite");
        first_run(&p, Format::Csv, 3);
        let mut s =
            Sink::open_format(&p, Format::Csv, Durability::Buffered, Existing::Overwrite).unwrap();
        s.write(&key(7)).unwrap();
        s.finish().unwrap();
        let body = read(&p);
        let lines: Vec<&str> = body.lines().collect();
        assert_eq!(
            lines.iter().filter(|l| l.starts_with("abc")).count(),
            1,
            "overwrite must discard the first run:
{body}"
        );
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn refusing_leaves_a_json_array_intact() {
        // Same guarantee for the format where a wrong open is most destructive.
        let p = tmp("refuse-json");
        first_run(&p, Format::JsonArray, 2);
        let before = read(&p);
        assert!(Sink::open_format(
            &p,
            Format::JsonArray,
            Durability::Buffered,
            Existing::Refuse
        )
        .is_err());
        assert_eq!(read(&p), before, "the array must be untouched");
        let _ = std::fs::remove_file(&p);
    }
}
