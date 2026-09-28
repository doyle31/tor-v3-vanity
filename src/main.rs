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
/// losing keys.
pub fn cuda_try_loop(
    prefixes: &[String],
    sender: crossbeam_channel::Sender<[u8; 64]>,
    tries_sender: crossbeam_channel::Sender<u64>,
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

            let mut byte_prefixes_owned: Vec<_> = prefixes
                .into_iter()
                .map(|a| BytePrefixOwned::from_str(&a))
                .collect();
            let mut byte_prefixes: Vec<_> = byte_prefixes_owned
                .iter_mut()
                .map(|bp| bp.as_byte_prefix())
                .collect();
            let mut gpu_byte_prefixes = DeviceBuffer::from_slice(&byte_prefixes).unwrap();

            let mut params = DeviceBox::new(&core::KernelParams {
                seed: gpu_seed.as_device_ptr(),
                byte_prefixes: gpu_byte_prefixes.as_device_ptr(),
                byte_prefixes_len: gpu_byte_prefixes.len(),
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

            let threads = *[
                fn_max_threads,
                gpu_max_threads.min(core::BLOCK as u32),
                gpu_max_registers / fn_registers.max(1),
            ]
            .iter()
            .min()
            .unwrap();
            let blocks = (gpu_cores * gpu_max_threads / threads).max(1);

            // Each launch examines BATCH keys per thread.
            let keys_per_launch = threads as u64 * blocks as u64 * core::BATCH as u64;

            println!(
                "Launching kernel on device #{} with {} threads and {} blocks \
                 ({} keys per launch)",
                i, threads, blocks, keys_per_launch
            );

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
                        sender.send(out).unwrap();
                    }
                }

                tries_sender.send(keys_per_launch).unwrap();
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

fn main() {
    let app = clap::App::new("t3v")
        .arg(
            clap::Arg::with_name("PREFIX")
                .required(true)
                .multiple(true)
                .value_delimiter(",")
                .help("Desired prefix"),
        )
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
        );
    let matches = app.get_matches();

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
    let dropped_keys = cuda_try_loop(&prefixes, send, send_tries).unwrap();

    let dropped_keys = dropped_keys.clone();
    std::thread::spawn(move || {
        let now = Instant::now();
        let mut last_log = Instant::now();
        let mut tries = 0_f64;
        let expected = 2_f64.powi(5 * max_len as i32);
        loop {
            tries += recv_tries.recv().unwrap() as f64;
            let dur = now.elapsed().as_secs_f64();
            let dur_pretty =
                PrettyDur(chrono::Duration::from_std(Duration::from_secs_f64(dur)).unwrap());
            let progress = tries / expected;
            let expected_dur = dur / progress;
            let expected_dur_pretty = PrettyDur(
                chrono::Duration::from_std(Duration::from_secs_f64(expected_dur)).unwrap(),
            );
            if last_log.elapsed() > Duration::from_secs(30) {
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
