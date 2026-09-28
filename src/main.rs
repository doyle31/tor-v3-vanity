use rand::RngCore;
use std::path::PathBuf;
use std::time::Duration;
use std::time::Instant;

use failure::Error;
use sha3::{Digest, Sha3_256};
use tor_v3_vanity_core as core;

pub mod output;

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

pub struct BytePrefixOwned {
    pub byte_prefix: rustacuda::memory::DeviceBuffer<u8>,
    pub last_byte_idx: usize,
    pub last_byte_mask: u8,
    /// Capture ring: `SLOTS` expanded keys of `SLOT_BYTES` bytes each.
    pub out: rustacuda::memory::DeviceBuffer<u8>,
    /// How many keys the kernel published into the ring this launch.
    pub count: rustacuda::memory::DeviceBox<u32>,
}
impl BytePrefixOwned {
    pub fn from_str(s: &str) -> Self {
        let byte_prefix = base32::decode(
            base32::Alphabet::RFC4648 { padding: false },
            &format!("{}aa", s),
        )
        .expect("prefix must be base32");
        let mut last_byte_idx = 5 * s.len() / 8 - 1;
        let n_bits = (5 * s.len()) % 8;
        let last_byte_mask = ((1 << n_bits) - 1) << (8 - n_bits);
        if last_byte_mask > 0 {
            last_byte_idx += 1;
        }
        let gpu_byte_prefix = rustacuda::memory::DeviceBuffer::from_slice(&byte_prefix).unwrap();
        // The kernel publishes into a ring of `SLOTS` expanded keys per prefix,
        // not a single slot: one slot capped a whole launch at one key.
        let out = vec![0u8; core::SLOTS * core::SLOT_BYTES];
        let gpu_out = rustacuda::memory::DeviceBuffer::from_slice(&out).unwrap();
        let gpu_count = rustacuda::memory::DeviceBox::new(&0u32).unwrap();
        BytePrefixOwned {
            byte_prefix: gpu_byte_prefix,
            last_byte_idx,
            last_byte_mask,
            out: gpu_out,
            count: gpu_count,
        }
    }
    pub fn as_byte_prefix(&mut self) -> core::BytePrefix {
        core::BytePrefix {
            byte_prefix: self.byte_prefix.as_device_ptr(),
            byte_prefix_len: self.byte_prefix.len(),
            last_byte_idx: self.last_byte_idx,
            last_byte_mask: self.last_byte_mask,
            out: self.out.as_device_ptr(),
            count: self.count.as_device_ptr(),
        }
    }
}

fn assert_crypto_rng<Rng: rand::CryptoRng>(rng: Rng) -> Rng {
    rng
}

/// Build the `[2^i]B` ladder on the host and upload it once per device.
///
/// The kernel needs this table to avoid recomputing the doubling ladder for
/// every candidate key. It is 255 * 4 * 5 u64s = 40800 bytes.
fn upload_power_table() -> Result<rustacuda::memory::DeviceBuffer<u64>, Error> {
    use core::group::{build_power_table, Ge, TABLE_LEN};

    let zero = Ge {
        x: [0; 5],
        y: [0; 5],
        z: [0; 5],
        t: [0; 5],
    };
    let mut table = [zero; TABLE_LEN];
    build_power_table(&mut table);

    // Reinterpret the point array as u64 words for upload. `Ge` is five-limb
    // field elements in u64 words, so this is a plain byte view with no
    // padding (all fields are [u64; 5] and the struct is 4 of them).
    let words: &[u64] = unsafe {
        std::slice::from_raw_parts(table.as_ptr() as *const u64, core::group::TABLE_LEN * 20)
    };
    Ok(rustacuda::memory::DeviceBuffer::from_slice(words).unwrap())
}

/// Start one search thread per CUDA device.
///
/// Returns a handle to the count of keys that matched but could not be
/// captured, so the caller can report a saturated ring instead of silently
/// losing keys. When `stop` is set the launch loops exit after the current
/// launch, which is how `--bench` shuts the GPU down cleanly.
pub fn cuda_try_loop(
    prefixes: &[String],
    sender: crossbeam_channel::Sender<[u8; 64]>,
    tries_sender: crossbeam_channel::Sender<u64>,
    stop: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
    verbose: bool,
    threads_override: Option<u32>,
    blocks_per_sm: Option<u32>,
) -> Result<std::sync::Arc<std::sync::atomic::AtomicU64>, Error> {
    use rustacuda::launch;
    use rustacuda::memory::DeviceBox;
    use rustacuda::prelude::*;
    use std::ffi::CString;

    rustacuda::init(CudaFlags::empty())?;
    // Keys found but not captured, because a launch published more matches
    // than one prefix's ring can hold. Surfaced so a saturated ring is visible
    // rather than silently discarding keys.
    let dropped_keys = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
    let mut i = 0;
    for device in rustacuda::device::Device::devices()? {
        let device = device?;
        let prefixes = prefixes.to_owned();
        let sender = sender.clone();
        let tries_sender = tries_sender.clone();
        let dropped_keys = dropped_keys.clone();
        let stop = stop.clone();
        std::thread::spawn(move || {
            use rand::RngCore;
            let mut csprng = assert_crypto_rng(rand::thread_rng());
            let _context =
                Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)
                    .unwrap();

            // Load PTX module
            let module_data = CString::new(include_str!(env!("KERNEL_PTX_PATH"))).unwrap();
            let kernel = Module::load_from_string(&module_data).unwrap();
            let function = kernel
                .get_function(std::ffi::CStr::from_bytes_with_nul(b"render\0").unwrap())
                .unwrap();

            let mut table = upload_power_table().unwrap();

            // Create a stream to submit work to
            let stream = Stream::new(StreamFlags::NON_BLOCKING, None).unwrap();

            // Move seed and prefix to device
            let mut seed = [0u8; 32];
            let mut gpu_seed = DeviceBuffer::from_slice(&seed).unwrap();
            // Placeholder cursor for the no-prefix benchmark path; never read
            // when there are no prefixes to publish into.
            let mut gpu_count_zero = DeviceBox::new(&0u32).unwrap();

            // An empty prefix list is the benchmark case: the kernel then skips
            // matching entirely. `DeviceBuffer::from_slice` cannot allocate a
            // zero-length buffer, and `copy_to` demands the readback buffer
            // match the upload exactly, so both sides carry one placeholder
            // entry. The kernel is told the real count -- 0 -- so it never
            // dereferences the placeholder, and the drain loop iterates
            // `byte_prefixes_owned`, which is empty.
            let mut byte_prefixes_owned: Vec<_> = prefixes
                .iter()
                .map(|a| BytePrefixOwned::from_str(a))
                .collect();
            let mut byte_prefixes: Vec<_> = byte_prefixes_owned
                .iter_mut()
                .map(|bp| bp.as_byte_prefix())
                .collect();
            let real_len = byte_prefixes.len();
            if byte_prefixes.is_empty() {
                byte_prefixes.push(core::BytePrefix {
                    byte_prefix: gpu_seed.as_device_ptr(),
                    byte_prefix_len: 0,
                    last_byte_idx: 0,
                    last_byte_mask: 0,
                    out: gpu_seed.as_device_ptr(),
                    count: gpu_count_zero.as_device_ptr(),
                });
            }
            let mut gpu_byte_prefixes = DeviceBuffer::from_slice(&byte_prefixes).unwrap();

            let mut params = DeviceBox::new(&core::KernelParams {
                seed: gpu_seed.as_device_ptr(),
                byte_prefixes: gpu_byte_prefixes.as_device_ptr(),
                byte_prefixes_len: real_len,
            })
            .unwrap();

            let fn_max_threads = function
                .get_attribute(rustacuda::function::FunctionAttribute::MaxThreadsPerBlock)
                .unwrap() as u32;
            let fn_registers = function
                .get_attribute(rustacuda::function::FunctionAttribute::NumRegisters)
                .unwrap() as u32;
            let gpu_max_threads = device
                .get_attribute(rustacuda::device::DeviceAttribute::MaxThreadsPerBlock)
                .unwrap() as u32;
            let gpu_max_registers = device
                .get_attribute(rustacuda::device::DeviceAttribute::MaxRegistersPerBlock)
                .unwrap() as u32;
            let gpu_cores = device
                .get_attribute(rustacuda::device::DeviceAttribute::MultiprocessorCount)
                .unwrap() as u32;
            // Spilled registers and the chunk buffers both land in local memory,
            // so this is the number that says whether the kernel is thrashing.
            let fn_local_bytes = function
                .get_attribute(rustacuda::function::FunctionAttribute::LocalSizeBytes)
                .unwrap_or(0) as u32;

            // The kernel's register count is the real limit: a kernel at 255
            // registers/thread allows only 8 warps per SM, so the "obvious" 256
            // threads per block is chosen for us by the hardware anyway.
            let by_occupancy = gpu_max_registers / fn_registers.max(1);
            let threads = match threads_override {
                Some(t) => t.min(fn_max_threads).min(gpu_max_threads),
                None => *[
                    fn_max_threads,
                    gpu_max_threads.min(core::BLOCK as u32),
                    by_occupancy,
                ]
                .iter()
                .min()
                .unwrap(),
            };
            let max_threads_per_sm = device
                .get_attribute(rustacuda::device::DeviceAttribute::MaxThreadsPerMultiprocessor)
                .unwrap_or(1536) as u32;
            let blocks = match blocks_per_sm {
                Some(p) => p * gpu_cores,
                None => (gpu_cores * max_threads_per_sm / threads).max(1),
            };

            // Each launch examines BATCH keys per thread.
            let keys_per_launch = threads as u64 * blocks as u64 * core::BATCH as u64;

            if verbose {
                // Occupancy is the number that decides whether register pressure
                // is hurting us: registers/thread against the 64K register file
                // per SM, and how many warps that allows.
                let regs_per_thread = fn_registers.max(1);
                let warps_by_regs = gpu_max_registers / (regs_per_thread * 32);
                println!(
                    "device {}: {} threads x {} blocks = {} warps/SM \
                     ({} SMs, {} keys/launch)\n  \
                     {} registers/thread -> {} warps/SM by registers; \
                     local memory {} B/thread; matching {}",
                    i,
                    threads,
                    blocks,
                    (threads / 32) * blocks / gpu_cores.max(1),
                    gpu_cores,
                    keys_per_launch,
                    regs_per_thread,
                    warps_by_regs,
                    fn_local_bytes,
                    if real_len == 0 { "OFF" } else { "on" },
                );
            }

            // Staging buffer for the capture ring, allocated once and reused:
            // `copy_to` insists on a full-length destination.
            let mut ring = vec![0u8; core::SLOTS * core::SLOT_BYTES];

            loop {
                csprng.fill_bytes(&mut seed);
                gpu_seed.copy_from(&seed).unwrap();
                unsafe {
                    launch!(kernel.render<<<blocks, threads, 0, stream>>>(
                        params.as_device_ptr(),
                        table.as_device_ptr()
                    ))
                    .unwrap();
                }

                stream.synchronize().unwrap();
                gpu_byte_prefixes.copy_to(&mut byte_prefixes).unwrap();

                for prefix in &mut byte_prefixes_owned {
                    // Drain every key the kernel captured this launch, not just
                    // one. `count` is the true number of matches; the ring can
                    // only hold SLOTS of them, and any excess was dropped
                    // rather than overwritten.
                    let mut count: u32 = 0;
                    prefix.count.copy_to(&mut count).unwrap();
                    if count == 0 {
                        continue;
                    }
                    prefix.count.copy_from(&0u32).unwrap();

                    let found = count as usize;
                    let captured = core::SLOTS.min(found);
                    if found > core::SLOTS {
                        dropped_keys.fetch_add(
                            (found - core::SLOTS) as u64,
                            std::sync::atomic::Ordering::Relaxed,
                        );
                    }
                    // `copy_to` requires equal lengths and rustacuda has no
                    // sub-range copy, so the whole ring comes back and only the
                    // captured prefix is used. At SLOTS=2048 that is 128 KiB per
                    // launch against a ~34 ms launch: ~0.04% of the budget.
                    prefix.out.copy_to(&mut ring).unwrap();
                    for k in 0..captured {
                        let mut out = [0u8; 64];
                        out.copy_from_slice(
                            &ring[k * core::SLOT_BYTES..(k + 1) * core::SLOT_BYTES],
                        );
                        // A disconnected receiver just means the consumer
                        // went away (shutdown, or a killed benchmark run).
                        let _ = sender.send(out);
                    }
                }

                let _ = tries_sender.send(keys_per_launch);

                if let Some(flag) = &stop {
                    if flag.load(std::sync::atomic::Ordering::Relaxed) {
                        break;
                    }
                }
            }
        });
        i += 1;
    }
    if i == 0 {
        eprintln!("No cuda devices available.");
        std::process::exit(2);
    }
    Ok(dropped_keys)
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

/// Benchmark raw kernel throughput: keys examined per second with no output
/// path and a prefix that can never match, so nothing but the GPU is measured.
///
/// Returns `(keys examined, seconds)`. Counts launches on the host, which is
/// exact: the tool advances by a fixed number of keys per launch.
fn bench_once(
    prefixes: &[String],
    secs: u64,
    verbose: bool,
    threads_override: Option<u32>,
    blocks_per_sm: Option<u32>,
) -> (u64, f64) {
    use crossbeam_channel::bounded;
    // The counter channel is bounded(1), so it must be drained continuously or
    // the launch loop blocks on send and the measurement is of the channel, not
    // the GPU. A dedicated thread accumulates; the main thread only samples.
    let (send_tries, recv_tries) = bounded::<u64>(1);
    let total = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
    let acc = total.clone();
    std::thread::spawn(move || {
        while let Ok(n) = recv_tries.recv() {
            acc.fetch_add(n, std::sync::atomic::Ordering::Relaxed);
        }
    });

    let (send, recv) = crossbeam_channel::bounded::<[u8; 64]>(4096);
    std::thread::spawn(move || while recv.recv().is_ok() {});

    // Each rep must fully stop the previous rep's launch loop, or two kernels
    // share the GPU and every run after the first is measured under contention.
    let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    cuda_try_loop(
        prefixes,
        send,
        send_tries,
        Some(stop.clone()),
        verbose,
        threads_override,
        blocks_per_sm,
    )
    .unwrap();

    let start = Instant::now();
    std::thread::sleep(Duration::from_secs(secs));
    let elapsed = start.elapsed().as_secs_f64();
    let keys = total.load(std::sync::atomic::Ordering::Relaxed);

    stop.store(true, std::sync::atomic::Ordering::Relaxed);
    // Let the launch loop finish its in-flight launch and exit before returning,
    // so the next rep starts on an idle GPU.
    std::thread::sleep(Duration::from_millis(500));
    (keys, elapsed)
}

/// Benchmark end-to-end for a prefix of length `len`, reporting keys *found*
/// per second -- the rate that determines how long a real search takes.
///
/// This exercises the whole path: capture ring, readback, host-side key
/// derivation, prefix verification and the output sink. A random prefix of the
/// given length is used, so the expected yield is one key per `2^(5*len)`.
fn bench_prefix_len(len: usize, secs: u64, reps: usize, verbose: bool) -> i32 {
    const ALPHABET: &[u8] = b"abcdefghijklmnopqrstuvwxyz234567";
    let mut best = 0f64;
    let mut best_expected = 0f64;

    for rep in 1..=reps {
        // A different prefix each run, so one unlucky prefix cannot flatter or
        // spoil the result.
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

        // Poll the output file: counting lines is the ground truth for how many
        // keys actually arrived, including any the host had to verify.
        let mut counted = 0u64;
        while start.elapsed().as_secs() < secs {
            std::thread::sleep(Duration::from_millis(500));
            counted = std::fs::read_to_string(&out)
                .map(|s| s.lines().filter(|l| !l.starts_with("onion,")).count() as u64)
                .unwrap_or(0);
        }
        let elapsed = start.elapsed().as_secs_f64();
        let _ = child.kill();
        let _ = child.wait();

        let rate = counted as f64 / elapsed;
        // Compare against the theoretical rate implied by raw throughput.
        let ratio = rate * expected;
        best = best.max(rate);
        best_expected = best_expected.max(ratio);
        if reps > 1 {
            println!(
                "  run {}/{}: prefix {} -> {:.2} keys found/s \
                 (raw-implied {:.0}/s, {:.1}% of ideal)",
                rep,
                reps,
                prefix,
                rate,
                ratio,
                100.0 * rate / ratio.max(1e-9)
            );
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    println!("{:.3}", best);
    eprintln!(
        "prefix length {}: {:.3} keys found/s (best of {})",
        len, best, reps
    );
    0
}

fn main() {
    let app = clap::App::new("t3v")
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
                .help("Check candidates against the prefix (default). \
                       The kernel skips compression and the batch inversion \
                       entirely when this is off"),
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
                       Implies --no-match, since measuring the matcher needs a \
                       prefix; pass --match to benchmark with matching on"),
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
                .help("Blocks per SM to launch. Default: fill the device. Fewer can \
                       help when a kernel is latency- rather than throughput-bound"),
        )
        .arg(
            clap::Arg::with_name("slots")
                .long("slots")
                .takes_value(true)
                .help("Keys captured per prefix per launch. Raise it if the run \
                       reports dropped keys (default 2048)"),
        );
    let matches = app.get_matches();

    // --- resolve every option explicitly, so nothing is implied by a filename
    // --- or by the presence of another flag.
    let benchmarking = matches.is_present("bench") || matches.is_present("bench-prefix");
    // Matching is on by default. `--bench` turns it off to isolate the kernel,
    // and `--match` puts it back on, so `--bench --match` measures the full path.
    let match_enabled = if matches.is_present("no-match") {
        false
    } else if matches.is_present("match") {
        true
    } else {
        !benchmarking
    };
    let secs: u64 = matches
        .value_of("bench")
        .map(|s| s.parse().expect("--bench takes a number of seconds"))
        .unwrap_or(30);
    let reps: usize = match matches.value_of("reps") {
        Some(r) => r.parse().expect("--reps takes a count"),
        None if matches.is_present("quick") => 1,
        None => 3,
    };
    let log_interval = match matches.value_of("log-interval") {
        Some(v) => Duration::from_secs(v.parse().expect("--log-interval takes seconds")),
        None => Duration::from_secs(30),
    };
    let verbose = matches.is_present("verbose");
    // Launch geometry is genuinely a runtime choice -- unlike CHUNK and BATCH,
    // which must stay compile-time constants to size the chunk buffers.
    let threads_override: Option<u32> = matches
        .value_of("threads")
        .map(|t| t.parse().expect("--threads takes a count"));
    let blocks_per_sm: Option<u32> = matches
        .value_of("blocks-per-sm")
        .map(|p| p.parse().expect("--blocks-per-sm takes a count"));

    // `--bench-prefix N` takes precedence: it measures the end-to-end rate for a
    // prefix of length N, which is the number that decides how long a real
    // search takes.
    if let Some(len) = matches.value_of("bench-prefix") {
        let len: usize = len
            .parse()
            .unwrap_or_else(|_| panic!("--bench-prefix takes a length, got {:?}", len));
        assert!((1..=12).contains(&len), "--bench-prefix must be 1..=12");
        std::process::exit(bench_prefix_len(len, secs, reps, verbose));
    }

    // Benchmark mode.
    if benchmarking {
        // With matching off the kernel generates keys and stops: no
        // compression, no batch inversion, no second pass. With `--match` the
        // caller-supplied prefix is used, which measures the full path.
        let prefixes: Vec<String> = if match_enabled {
            match matches.value_of("PREFIX") {
                Some(p) => vec![p.to_string()],
                None => vec!["z".to_string()],
            }
        } else {
            vec![]
        };
        let mut best = 0f64;
        for rep in 1..=reps {
            let (keys, elapsed) = bench_once(&prefixes, secs, verbose, threads_override, blocks_per_sm);
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
        println!("{:.3}", best / 1e6);
        std::process::exit(0);
    }

    let max_len = matches
        .values_of("PREFIX")
        .unwrap()
        .fold(0, |acc, x| std::cmp::max(acc, x.len()));
    let prefixes: Vec<_> = matches
        .values_of("PREFIX")
        .unwrap()
        .map(|a| a.to_string())
        .collect();

    // An explicit --format always wins. The file extension is only a fallback,
    // so `--out keys` and `--out keys.dat` both work.
    let format = matches.value_of("format").map(|name| {
        output::Format::parse(name)
            .unwrap_or_else(|| panic!("unknown --format {:?}; expected one of {}", name,
                                      output::Format::NAMES.join(", ")))
    });
    if format.is_some() && matches.value_of("out").is_none() {
        eprintln!("warning: --format has no effect without --out; using per-key files");
    }

    // One of two sinks: a single consolidated file, or a directory of per-key
    // Tor key files.
    let mut sink = match matches.value_of("out") {
        Some(p) => {
            let path = PathBuf::from(p);
            let chosen = format.unwrap_or_else(|| output::Format::from_path(&path));
            eprintln!("Writing matches to {} ({})", path.display(), chosen.name());
            output::Sink::open_format(&path, chosen).unwrap()
        }
        None => {
            let dst = matches
                .value_of("dst")
                .map(PathBuf::from)
                .unwrap_or_else(|| std::env::current_dir().unwrap());
            assert!(dst.is_dir(), "dst must be a directory");
            output::Sink::dir(&dst)
        }
    };

    let (send, recv) = crossbeam_channel::unbounded();
    let (send_tries, recv_tries) = crossbeam_channel::bounded(1);
    let dropped_keys =
        cuda_try_loop(
            &prefixes,
            send,
            send_tries,
            None,
            verbose,
            threads_override,
            blocks_per_sm,
        )
        .unwrap();

    let dropped_keys = dropped_keys.clone();
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
            let dur_pretty = PrettyDur(to_chrono(dur));
            let progress = tries / expected;
            let expected_dur = dur / progress;
            let expected_dur_pretty = PrettyDur(to_chrono(expected_dur));
            if log_interval > Duration::from_secs(0) && last_log.elapsed() > log_interval {
                println!("Tried {:.0} / {:.0} (expected) keys.", tries, expected);
                println!(
                    "Running for {} / {} (expected).",
                    dur_pretty, expected_dur_pretty
                );
                let dropped = dropped_keys.load(std::sync::atomic::Ordering::Relaxed);
                if dropped > 0 {
                    println!(
                        "WARNING: {} keys matched but were dropped -- a launch \
                         published more than the {} slots per prefix.",
                        dropped,
                        core::SLOTS
                    );
                }
                last_log = Instant::now();
            }
        }
    });

    loop {
        // The kernel returns the 64-byte expanded key, so the pubkey comes
        // straight from it with no host-side key derivation.
        let expanded = recv.recv().unwrap();
        let esk = ed25519_dalek::ExpandedSecretKey::from_bytes(&expanded[..]).unwrap();
        let pk: ed25519_dalek::PublicKey = (&esk).into();
        let onion = pubkey_to_onion(pk.as_bytes());

        // Several threads can match the same prefix in one launch, and they
        // all write to that prefix's single `out` slot, so the key read back
        // may not be the one that matched. Verify before emitting it; a
        // non-match just means we lost the race, and the search continues.
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
}
