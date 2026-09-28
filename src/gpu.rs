//! CUDA device management and the launch loop.
//!
//! One search thread per device. Each re-randomises the block seed, launches
//! the kernel, and drains the capture ring back to the host.
//!
//! The launch geometry is the one genuinely runtime-tunable part of the search.
//! `CHUNK` and `BATCH` in the device crate are not: they size local arrays and
//! drive loop unrolling, so they are compile-time constants, overridable only
//! at build time via `T3V_CHUNK` / `T3V_BATCH`.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use anyhow::Result;
use tor_v3_vanity_core as core;

/// Default threads per block, from `kernel.conf` via `build.rs`.
///
/// Deliberately *not* `core::BLOCK`. The device crate is a separate
/// compilation, and `option_env!` inside it resolves when that crate is
/// compiled -- `cargo:rustc-env` from this crate's build script does not reach
/// a dependency's compilation. So the host read the dependency's default of 256
/// while the kernel was built with the file's 128, and the setting silently
/// did nothing. This constant is set for the host's own compilation, and both
/// values come from the same file.
pub const DEFAULT_BLOCK: usize = match option_env!("T3V_BLOCK") {
    Some(s) => parse_usize(s.as_bytes()),
    None => 256,
};

/// Compile-time decimal parse, mirroring the device crate's own.
const fn parse_usize(bytes: &[u8]) -> usize {
    let mut out: usize = 0;
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i];
        if c < b'0' || c > b'9' {
            break;
        }
        out = out * 10 + (c - b'0') as usize;
        i += 1;
    }
    out
}

/// Count of keys that matched but could not be captured, shared with the
/// caller so a saturated ring is reported rather than silently losing keys.
pub type Dropped = Arc<AtomicU64>;

/// Everything the launch loop needs to know that is not per-device state.
#[derive(Clone, Copy, Default)]
pub struct Launch {
    pub verbose: bool,
    pub threads: Option<u32>,
    pub blocks_per_sm: Option<u32>,
}

/// A prefix to match against, plus the device-side capture ring it publishes
/// into.
pub struct BytePrefixOwned {
    pub byte_prefix: rustacuda::memory::DeviceBuffer<u8>,
    pub last_byte_idx: usize,
    pub last_byte_mask: u8,
    /// The same prefix bytes packed little-endian, which is what the kernel
    /// actually compares against.
    pub prefix_word: u64,
    /// Capture ring: `SLOTS` expanded keys of `SLOT_BYTES` bytes each.
    pub out: rustacuda::memory::DeviceBuffer<u8>,
    /// How many keys the kernel published into the ring this launch.
    pub count: rustacuda::memory::DeviceBox<u32>,
}

impl BytePrefixOwned {
    pub fn from_str(s: &str) -> Self {
        // The kernel tests the low eight bytes of the encoding in one 64-bit
        // word (`BytePrefix::matches_low8`), so a prefix that reaches past byte
        // 7 cannot be tested at all. `MAX_PREFIX_CHARS` is 12, i.e. 60 bits --
        // a 12-character prefix is already a 2^60-candidate search, so nothing
        // longer is worth finding. Rejecting it here is better than letting the
        // kernel silently never match.
        assert!(
            s.len() <= core::BytePrefix::MAX_PREFIX_CHARS,
            "a prefix may be at most {} characters, got {} ({:?})",
            core::BytePrefix::MAX_PREFIX_CHARS,
            s.len(),
            s
        );
        // `Rfc4648Lower`, not `Rfc4648`: base32 0.5 made the two alphabets
        // case-specific, and `Rfc4648` now rejects lowercase outright with a
        // `None` rather than decoding it. Prefixes reach here already
        // lowercased by `cli::parse`, so this is the matching alphabet. With
        // the uppercase one every prefix decode returns `None` and every
        // search silently finds nothing.
        let byte_prefix = base32::decode(
            base32::Alphabet::Rfc4648Lower { padding: false },
            &format!("{}aa", s),
        )
        .expect("prefix must be base32");
        // Which bytes of the encoding a prefix of `s` actually constrains, and
        // how much of the final one.
        //
        // `5 * len` bits are pinned, so they span `ceil(5 * len / 8)` bytes and
        // the last of those is the only one that may be partial. That is the
        // last byte index. The previous spelling, `5 * len / 8 - 1` followed by
        // an increment when the mask was nonzero, is the same value for every
        // length except one: at `len == 1` it computes `0 - 1` and underflows
        // a `usize`. A release build only got the right answer because the
        // underflow wrapped to `usize::MAX` and the increment wrapped it back
        // to `0`; a debug build, and any caller that does arithmetic on it, sees
        // a panic. `div_ceil` states the intent and has no wraparound to rely
        // on -- the prefix length is already known to be at least 1, so the
        // result is 0 or greater.
        let bits = 5 * s.len();
        let n_bits = bits % 8;
        let last_byte_idx = bits.div_ceil(8) - 1;
        // When 5*len is a whole number of bytes (len 8, 16, ...) the last byte is
        // used in full, so the mask must select all 8 of its bits. Leaving it
        // zero degenerates the comparison to `0 == slice[i]`, which holds for
        // almost no real prefix -- so an 8-character prefix matched nothing,
        // ever, and a search for one would run forever finding no keys.
        let last_byte_mask = if n_bits == 0 {
            0xFF
        } else {
            ((1 << n_bits) - 1) << (8 - n_bits)
        };
        debug_assert!(last_byte_idx < 8);
        let prefix_word = core::BytePrefix::pack_prefix_word(&byte_prefix);
        let gpu_byte_prefix = rustacuda::memory::DeviceBuffer::from_slice(&byte_prefix).unwrap();
        let out = vec![0u8; core::SLOTS * core::SLOT_BYTES];
        let gpu_out = rustacuda::memory::DeviceBuffer::from_slice(&out).unwrap();
        let gpu_count = rustacuda::memory::DeviceBox::new(&0u32).unwrap();
        BytePrefixOwned {
            byte_prefix: gpu_byte_prefix,
            last_byte_idx,
            last_byte_mask,
            prefix_word,
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
            prefix_word: self.prefix_word,
            out: self.out.as_device_ptr(),
            count: self.count.as_device_ptr(),
        }
    }
}

fn assert_crypto_rng<Rng: rand::CryptoRng>(rng: Rng) -> Rng {
    rng
}

/// Build the base-point tables on the host and upload them once per device.
///
/// The kernel needs these to avoid recomputing the base-point ladder for every
/// candidate key. `BaseTables` is the fixed-window comb over the scalar's bits
/// (every window's subset sums, in Niels form) plus the walk step `8B`.
fn upload_base_tables() -> Result<rustacuda::memory::DeviceBuffer<u64>> {
    use core::group::{build_comb_table, BaseTables, BASE_TABLE_WORDS};

    // `BaseTables` is `#[repr(C)]` and contains nothing but `[u64; 5]`, so it has
    // no padding and a flat u64 view of it is exact -- which
    // `base_tables_layout_is_flat` in the core crate checks.
    let mut base: BaseTables = unsafe { std::mem::zeroed() };
    build_comb_table(&mut base.comb, &mut base.eight);

    let words: &[u64] =
        unsafe { std::slice::from_raw_parts(&base as *const BaseTables as *const u64, BASE_TABLE_WORDS) };
    Ok(rustacuda::memory::DeviceBuffer::from_slice(words).unwrap())
}

/// Start one search thread per CUDA device.
///
/// An empty `prefixes` is the benchmark case: the kernel then skips matching
/// entirely. When `stop` is set the loops exit after the current launch.
///
/// The returned handles must be joined before the process tears CUDA down.
/// Setting `stop` only *asks* a thread to finish its current launch; a thread
/// still in `stream.synchronize` when the process starts exiting gets
/// `Error::Deinitialized` from the driver, which used to panic the worker and
/// leave the search silently producing nothing. Joining is what makes the
/// shutdown deterministic.
pub fn start(
    prefixes: &[String],
    sender: crossbeam_channel::Sender<[u8; 64]>,
    tries_sender: crossbeam_channel::Sender<u64>,
    stop: Option<Arc<AtomicBool>>,
    launch: Launch,
) -> Result<(Dropped, Vec<std::thread::JoinHandle<()>>)> {
    use rustacuda::launch;
    use rustacuda::memory::DeviceBox;
    use rustacuda::prelude::*;
    use std::ffi::CString;

    rustacuda::init(CudaFlags::empty())?;
    let dropped_keys: Dropped = Arc::new(AtomicU64::new(0));
    let mut workers: Vec<std::thread::JoinHandle<()>> = Vec::new();
    let mut i = 0;
    for device in rustacuda::device::Device::devices()? {
        let device = device?;
        let prefixes = prefixes.to_owned();
        let sender = sender.clone();
        let tries_sender = tries_sender.clone();
        let dropped_keys = dropped_keys.clone();
        let stop = stop.clone();
        workers.push(std::thread::spawn(move || {
            use rand::RngCore;
            let mut csprng = assert_crypto_rng(rand::rng());
            let _context =
                Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)
                    .unwrap();

            // Re-install the signal handlers *here*, after the CUDA context
            // exists, and re-assert them on every launch below.
            //
            // Creating a CUDA context resets SIGTERM and SIGINT to the default,
            // so the install in `main` is silently discarded. `sigaction` is
            // process-wide, so installing from this worker is equivalent to
            // installing from main.
            //
            // Re-asserting per launch is not belt-and-braces, it is necessary.
            // The reset is not at a fixed point in the init sequence: two runs
            // of the same binary, sampled at the same point six seconds in, gave
            // SigCgt = 0x100004442 (both caught) and SigCgt = 0x10000 (neither),
            // so something in CUDA's setup or the PTX JIT clears them
            // nondeterministically and no single "install after X" placement is
            // reliable. Re-asserting once per launch bounds the window in which a
            // signal can be missed to one launch, and costs two syscalls per
            // ~70 ms.
            //
            // The symptom when this is missing is silent and bad: Ctrl-C or
            // `kill` kills the process outright, the shutdown path never runs,
            // and a `.json` array file is left without its closing bracket --
            // unparseable. Which is the exact failure `signal` exists to prevent.
            if let Err(e) = crate::signal::install() {
                eprintln!(
                    "warning: device {}: cannot install signal handlers after CUDA init: {}",
                    i, e
                );
            }

            let module_data = CString::new(include_str!(env!("KERNEL_PTX_PATH"))).unwrap();
            let kernel = Module::load_from_string(&module_data).unwrap();
            let function = kernel
                .get_function(std::ffi::CStr::from_bytes_with_nul(b"render\0").unwrap())
                .unwrap();

            let mut table = upload_base_tables().unwrap();
            let stream = Stream::new(StreamFlags::NON_BLOCKING, None).unwrap();

            let mut seed = [0u8; 32];
            let mut gpu_seed = DeviceBuffer::from_slice(&seed).unwrap();
            // Placeholder cursor for the no-prefix benchmark path; never read
            // when there are no prefixes to publish into.
            let mut gpu_count_zero = DeviceBox::new(&0u32).unwrap();

            let mut byte_prefixes_owned: Vec<_> =
                prefixes.iter().map(|a| BytePrefixOwned::from_str(a)).collect();
            let mut byte_prefixes: Vec<_> = byte_prefixes_owned
                .iter_mut()
                .map(|bp| bp.as_byte_prefix())
                .collect();
            // `DeviceBuffer::from_slice` cannot allocate a zero-length buffer,
            // and `copy_to` demands the readback buffer match the upload
            // exactly, so both sides carry one placeholder entry when there are
            // no prefixes. The kernel is told the real count -- 0 -- so it never
            // dereferences the placeholder, and the drain loop iterates
            // `byte_prefixes_owned`, which is empty.
            let real_len = byte_prefixes.len();
            if byte_prefixes.is_empty() {
                byte_prefixes.push(core::BytePrefix {
                    byte_prefix: gpu_seed.as_device_ptr(),
                    byte_prefix_len: 0,
                    last_byte_idx: 0,
                    last_byte_mask: 0,
                    prefix_word: 0,
                    out: gpu_seed.as_device_ptr(),
                    count: gpu_count_zero.as_device_ptr(),
                });
            }
            let mut gpu_byte_prefixes = DeviceBuffer::from_slice(&byte_prefixes).unwrap();

            let matching = real_len > 0;
            let (threads, blocks, keys_per_launch) =
                geometry(&device, &function, &launch);
            if launch.verbose {
                report_geometry(&device, &function, i, threads, blocks, keys_per_launch, matching);
            }

            // One u64 per thread, written by the no-prefix (`--bench`) path and
            // never read back. It exists so the walk is not dead code; see
            // `KernelParams::bench_sink` for why that exposure is worth closing
            // even though the walk is not currently being deleted.
            let mut bench_sink: DeviceBuffer<u64> =
                DeviceBuffer::from_slice(&vec![0u64; threads as usize * blocks as usize]).unwrap();

            let mut params = DeviceBox::new(&core::KernelParams {
                seed: gpu_seed.as_device_ptr(),
                byte_prefixes: gpu_byte_prefixes.as_device_ptr(),
                byte_prefixes_len: real_len,
                bench_sink: bench_sink.as_device_ptr(),
            })
            .unwrap();

            // Staging buffer for the capture ring, allocated once and reused:
            // `copy_to` insists on a full-length destination.
            let mut ring = vec![0u8; core::SLOTS * core::SLOT_BYTES];

            loop {
                // Re-assert the handlers every launch: CUDA clears them
                // nondeterministically during setup (see the note above). Cheap,
                // and it is what makes Ctrl-C deterministic.
                let _ = crate::signal::install();
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
                    // only hold SLOTS of them, and any excess was dropped rather
                    // than overwritten.
                    let mut count: u32 = 0;
                    prefix.count.copy_to(&mut count).unwrap();
                    if count == 0 {
                        continue;
                    }
                    prefix.count.copy_from(&0u32).unwrap();

                    let found = count as usize;
                    let captured = core::SLOTS.min(found);
                    if found > core::SLOTS {
                        dropped_keys.fetch_add((found - core::SLOTS) as u64, Ordering::Relaxed);
                    }
                    // `copy_to` requires equal lengths and rustacuda has no
                    // sub-range copy, so the whole ring comes back and only the
                    // captured prefix is used. At SLOTS=16384 that is 1 MiB per
                    // prefix per launch; at the measured ~14.5 launches/s and
                    // ~69 ms per launch that is ~15 MB/s, about 0.1% of a PCIe
                    // 3.0 x16 link and ~0.3% of the launch.
                    prefix.out.copy_to(&mut ring).unwrap();
                    for k in 0..captured {
                        let mut out = [0u8; 64];
                        out.copy_from_slice(
                            &ring[k * core::SLOT_BYTES..(k + 1) * core::SLOT_BYTES],
                        );
                        // A disconnected receiver just means the consumer went
                        // away: shutdown, or a killed benchmark run.
                        let _ = sender.send(out);
                    }
                }

                let _ = tries_sender.send(keys_per_launch);

                if let Some(flag) = &stop {
                    if flag.load(Ordering::Relaxed) {
                        break;
                    }
                }
            }
        }));
        i += 1;
    }
    if i == 0 {
        eprintln!("No cuda devices available.");
        std::process::exit(2);
    }
    Ok((dropped_keys, workers))
}

/// Choose threads per block and blocks per launch.
///
/// The kernel's register count is the real limit on threads per block: at 255
/// registers/thread a 64K register file admits only 257 threads per SM, so the
/// "obvious" 256 is chosen for us by the hardware anyway. Blocks are then sized
/// to fill the device's thread capacity, which on this GPU means 120 rather
/// than the 80 an earlier `MaxThreadsPerBlock`-based formula produced.
fn geometry(
    device: &rustacuda::device::Device,
    function: &rustacuda::function::Function,
    launch: &Launch,
) -> (u32, u32, u64) {
    use rustacuda::device::DeviceAttribute;
    use rustacuda::function::FunctionAttribute;

    let fn_max_threads = function
        .get_attribute(FunctionAttribute::MaxThreadsPerBlock)
        .unwrap() as u32;
    let fn_registers = function
        .get_attribute(FunctionAttribute::NumRegisters)
        .unwrap() as u32;
    let gpu_max_threads = device
        .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
        .unwrap() as u32;
    let gpu_max_registers = device
        .get_attribute(DeviceAttribute::MaxRegistersPerBlock)
        .unwrap() as u32;
    let gpu_cores = device
        .get_attribute(DeviceAttribute::MultiprocessorCount)
        .unwrap() as u32;
    let max_threads_per_sm = device
        .get_attribute(DeviceAttribute::MaxThreadsPerMultiprocessor)
        .unwrap_or(1536) as u32;

    let by_occupancy = gpu_max_registers / fn_registers.max(1);
    let threads = match launch.threads {
        Some(t) => t.min(fn_max_threads).min(gpu_max_threads),
        None => *[
            fn_max_threads,
            gpu_max_threads.min(DEFAULT_BLOCK as u32),
            by_occupancy,
        ]
        .iter()
        .min()
        .unwrap(),
    };
    let blocks = match launch.blocks_per_sm {
        Some(p) => p * gpu_cores,
        None => (gpu_cores * max_threads_per_sm / threads).max(1),
    };
    let keys_per_launch = threads as u64 * blocks as u64 * core::BATCH as u64;
    (threads, blocks, keys_per_launch)
}

fn report_geometry(
    device: &rustacuda::device::Device,
    function: &rustacuda::function::Function,
    index: u32,
    threads: u32,
    blocks: u32,
    keys_per_launch: u64,
    matching: bool,
) {
    use rustacuda::device::DeviceAttribute;
    use rustacuda::function::FunctionAttribute;

    let fn_registers = function
        .get_attribute(FunctionAttribute::NumRegisters)
        .unwrap() as u32;
    let fn_local_bytes = function
        .get_attribute(FunctionAttribute::LocalSizeBytes)
        .unwrap_or(0) as u32;
    let gpu_max_registers = device
        .get_attribute(DeviceAttribute::MaxRegistersPerBlock)
        .unwrap() as u32;
    let gpu_cores = device
        .get_attribute(DeviceAttribute::MultiprocessorCount)
        .unwrap() as u32;

    let regs = fn_registers.max(1);
    println!(
        "device {}: {} threads x {} blocks = {} warps/SM ({} SMs, {} keys/launch)",
        index,
        threads,
        blocks,
        (threads / 32) * blocks / gpu_cores.max(1),
        gpu_cores,
        keys_per_launch
    );
    println!(
        "  {} registers/thread -> {} warps/SM by registers; local memory {} B/thread; \
         matching {}",
        regs,
        gpu_max_registers / (regs * 32),
        fn_local_bytes,
        if matching { "on" } else { "OFF" }
    );
}

#[cfg(test)]
mod prefix_tests {
    //! The prefix comparison is the one piece of host-side logic with no
    //! independent oracle, and it was silently wrong for 8-character prefixes.
    //!
    //! `5*len` is a whole number of bytes when `len` is a multiple of 8, so the
    //! partial-byte mask came out as zero and the comparison collapsed to
    //! `0 == slice[4]`. Almost no prefix satisfies that, so an 8-character
    //! search found nothing, ever -- and a user waiting for a 2^40 search would
    //! never know why. The mask is now 0xFF in that case, and these tests
    //! exercise every length against the real base32 encoding.

    /// Reproduce the host-side comparison from `BytePrefixOwned::from_str`.
    fn compare(prefix: &str, data: &[u8; 32]) -> bool {
        let padded = format!("{}aa", prefix);
        let byte_prefix = base32::decode(
            base32::Alphabet::Rfc4648Lower { padding: false },
            &padded,
        )
        .expect("base32");
        // Must stay in step with the same computation in
        // `BytePrefixOwned::from_str`; this is a hand-copied stand-in for it
        // because that one needs a CUDA context to allocate its device buffers.
        let bits = 5 * prefix.len();
        let n_bits = bits % 8;
        let last_byte_idx = bits.div_ceil(8) - 1;
        let last_byte_mask = if n_bits == 0 {
            0xFF
        } else {
            ((1 << n_bits) - 1) << (8 - n_bits)
        };
        data.starts_with(&byte_prefix[..last_byte_idx])
            && data[last_byte_idx] & last_byte_mask == byte_prefix[last_byte_idx]
    }

    /// The bytes a prefix should match, as a pubkey would present them.
    fn expected_bytes(prefix: &str) -> [u8; 32] {
        let padded = format!("{}aa", prefix);
        let decoded = base32::decode(
            base32::Alphabet::Rfc4648Lower { padding: false },
            &padded,
        )
        .expect("base32");
        let mut out = [0u8; 32];
        out[..decoded.len()].copy_from_slice(&decoded);
        out
    }

    #[test]
    fn every_prefix_length_matches_its_own_bytes() {
        // The regression: 8 characters used to fail right here.
        for len in 1..=12usize {
            for filler in ["a", "z", "7", "m", "q"] {
                let prefix: String = filler.repeat(len);
                assert!(
                    compare(&prefix, &expected_bytes(&prefix)),
                    "prefix {:?} (len {}) should match its own bytes",
                    prefix,
                    len
                );
            }
        }
    }

    #[test]
    fn a_prefix_rejects_bytes_that_differ_within_the_masked_range() {
        // Guards the opposite failure: a test that passes by matching
        // everything. Only bits the comparison actually inspects may be
        // flipped -- for a 1-character prefix that is 5 bits of byte 0, so
        // perturbing byte 1 is *correctly* ignored.
        for len in 1..=12usize {
            let prefix = "a".repeat(len);
            let want = expected_bytes(&prefix);
            // Same computation as `BytePrefixOwned::from_str`: a prefix pins
            // `5 * len` bits, spanning `ceil(5 * len / 8)` bytes, the last of
            // which is partial unless `5 * len` is a whole number of bytes.
            //
            // This was previously written as `if mask != 0 { 5 * len / 8 } else
            // { 5 * len / 8 - 1 }`, which inverts the condition -- a full-width
            // mask is precisely the whole-byte case, so len 8 computed the last
            // index as 5 instead of 4. That then demanded the comparison reject
            // a flip in byte 5, which an 8-character prefix does not constrain,
            // so the assertion failed on a case that is supposed to match. The
            // test was wrong about the code, not the reverse.
            let bits = 5 * len;
            let n_bits = bits % 8;
            let mask: u8 = if n_bits == 0 {
                0xFF
            } else {
                ((1 << n_bits) - 1) << (8 - n_bits)
            };
            let last_idx = bits.div_ceil(8) - 1;
            for byte in 0..=last_idx {
                // Bytes before the last are compared in full; the last only
                // through its mask.
                let bits = if byte == last_idx { mask } else { 0xFF };
                for bit in 0..8u8 {
                    if bits & (1 << bit) == 0 {
                        continue;
                    }
                    let mut other = want;
                    other[byte] ^= 1 << bit;
                    assert!(
                        !compare(&prefix, &other),
                        "prefix {:?} must reject bit {} of byte {} flipped",
                        prefix,
                        bit,
                        byte
                    );
                }
            }
        }
    }

    #[test]
    fn eight_character_prefixes_are_distinguishable() {
        // The specific case that was broken, spelled out so the failure is
        // legible in the log rather than hidden inside a loop over lengths.
        let a = "a".repeat(8);
        let b = "b".repeat(8);
        assert!(compare(&a, &expected_bytes(&a)), "8 x 'a' must match itself");
        assert!(compare(&b, &expected_bytes(&b)), "8 x 'b' must match itself");
        assert!(!compare(&a, &expected_bytes(&b)), "8 x 'a' must reject 8 x 'b'");
    }

    #[test]
    fn mask_is_full_width_exactly_when_the_prefix_is_a_whole_number_of_bytes() {
        for len in 1..=12usize {
            let n_bits = (5 * len) % 8;
            let mask: u8 = if n_bits == 0 {
                0xFF
            } else {
                ((1 << n_bits) - 1) << (8 - n_bits)
            };
            if n_bits == 0 {
                assert_eq!(mask, 0xFF, "len {} uses whole bytes", len);
            } else {
                let used = mask.count_ones();
                assert_eq!(used as u32, n_bits as u32, "len {}", len);
            }
        }
    }
}
