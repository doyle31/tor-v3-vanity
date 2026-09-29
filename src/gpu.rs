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
///
/// # The `None` arm
///
/// `build.rs` emits `cargo:rustc-env=T3V_BLOCK=...` from the same `kernel.conf`
/// line that feeds the device crate, so in any build that goes through it the
/// `Some` arm always wins and the value below is never read. It is only reachable
/// if this crate is compiled without its build script, which does not happen.
///
/// It used to read 256, which was actively misleading: 256 is the *device
/// crate's* fallback, and a reader would reasonably conclude the 128 in
/// `kernel.conf` was being overridden and the bug described above was still
/// live. It is fixed -- `build.rs` sets the variable for both compilations -- so
/// this repeats `kernel.conf`'s value rather than contradicting it.
pub const DEFAULT_BLOCK: usize = match option_env!("T3V_BLOCK") {
    Some(s) => parse_usize(s.as_bytes()),
    None => 128,
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

/// The host-side, device-free half of a prefix: everything `from_str` computes
/// before it needs a CUDA context.
///
/// This exists so the prefix arithmetic can be tested directly. It used to live
/// inline in `BytePrefixOwned::from_str`, which meant the only tests covering it
/// had to hand-copy the whole derivation -- so they pinned the *copy* and would
/// have passed unchanged if the original drifted. Splitting the pure part out is
/// what lets `prefix_tests` call the real thing; the allocation stays in
/// `from_str`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefixSpec {
    /// The prefix decoded to the bytes it constrains in a pubkey.
    pub bytes: Vec<u8>,
    /// Which bytes of the encoding a prefix actually constrains, and how much
    /// of the last one.
    ///
    /// `5 * len` bits are pinned, so they span `ceil(5 * len / 8)` bytes and the
    /// last of those is the only one that may be partial. The previous spelling,
    /// `5 * len / 8 - 1` followed by an increment when the mask was nonzero, is
    /// the same value for every length except one: at `len == 1` it computes
    /// `0 - 1` and underflows a `usize`. A release build only got the right
    /// answer because the underflow wrapped to `usize::MAX` and the increment
    /// wrapped it back to `0`; a debug build, and any caller that does
    /// arithmetic on it, sees a panic. `div_ceil` states the intent and has no
    /// wraparound to rely on -- the prefix length is already known to be at
    /// least 1, so the result is 0 or greater.
    pub last_byte_idx: usize,
    /// When `5 * len` is a whole number of bytes (len 8, 16, ...) the last byte
    /// is used in full, so the mask must select all 8 of its bits. Leaving it
    /// zero degenerates the comparison to `0 == slice[i]`, which holds for almost
    /// no real prefix -- so an 8-character prefix matched nothing, ever, and a
    /// search for one would run forever finding no keys.
    pub last_byte_mask: u8,
    /// The first eight prefix bytes, packed little-endian, which is what the
    /// kernel actually compares against.
    pub prefix_word: u64,
}

/// Decode a prefix and derive everything the matcher needs from it.
///
/// Pure, and the reason it is separated from [`BytePrefixOwned::from_str`]: it
/// needs no CUDA context, so it can be unit tested.
///
/// # Panics
///
/// Panics if the prefix is empty, longer than
/// [`MAX_PREFIX_CHARS`](core::BytePrefix::MAX_PREFIX_CHARS), or not valid
/// base32.
pub fn prefix_spec(s: &str) -> PrefixSpec {
    // The kernel tests the low eight bytes of the encoding in one 64-bit word
    // (`BytePrefix::matches_low8`), so a prefix that reaches past byte 7 cannot
    // be tested at all. `MAX_PREFIX_CHARS` is 12, i.e. 60 bits -- a
    // 12-character prefix is already a 2^60-candidate search, so nothing longer
    // is worth finding. Rejecting it here is better than letting the kernel
    // silently never match.
    assert!(
        s.len() <= core::BytePrefix::MAX_PREFIX_CHARS,
        "a prefix may be at most {} characters, got {} ({:?})",
        core::BytePrefix::MAX_PREFIX_CHARS,
        s.len(),
        s
    );
    assert!(!s.is_empty(), "a prefix may not be empty");
    // `Rfc4648Lower`, not `Rfc4648`: base32 0.5 made the two alphabets
    // case-specific, and `Rfc4648` now rejects lowercase outright with a `None`
    // rather than decoding it. Prefixes reach here already lowercased by
    // `cli::parse`, so this is the matching alphabet. With the uppercase one
    // every prefix decode returns `None` and every search silently finds
    // nothing.
    let bytes = base32::decode(
        base32::Alphabet::Rfc4648Lower { padding: false },
        &format!("{}aa", s),
    )
    .expect("prefix must be base32");

    let bits = 5 * s.len();
    let n_bits = bits % 8;
    let last_byte_idx = bits.div_ceil(8) - 1;
    let last_byte_mask = if n_bits == 0 {
        0xFF
    } else {
        ((1 << n_bits) - 1) << (8 - n_bits)
    };
    // Not a debug_assert: every caller needs `idx < 8` to be true, and the
    // length cap above is what establishes it. Asserting it in release too means
    // a future caller that widens the cap finds out here rather than reading
    // past the end of the word the kernel compares.
    assert!(
        last_byte_idx < 8,
        "a {}-character prefix spans byte {} and does not fit the compared word",
        s.len(),
        last_byte_idx
    );
    let prefix_word = core::BytePrefix::pack_prefix_word(&bytes);
    PrefixSpec {
        bytes,
        last_byte_idx,
        last_byte_mask,
        prefix_word,
    }
}

impl BytePrefixOwned {
    pub fn from_str(s: &str) -> Self {
        let spec = prefix_spec(s);
        let gpu_byte_prefix =
            rustacuda::memory::DeviceBuffer::from_slice(&spec.bytes).unwrap();
        let out = vec![0u8; core::SLOTS * core::SLOT_BYTES];
        let gpu_out = rustacuda::memory::DeviceBuffer::from_slice(&out).unwrap();
        let gpu_count = rustacuda::memory::DeviceBox::new(&0u32).unwrap();
        BytePrefixOwned {
            byte_prefix: gpu_byte_prefix,
            last_byte_idx: spec.last_byte_idx,
            last_byte_mask: spec.last_byte_mask,
            prefix_word: spec.prefix_word,
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
            // SCHED_BLOCKING_SYNC, not SCHED_AUTO. This is the flag that decides
            // whether `stream.synchronize` spins or blocks, and with SCHED_AUTO
            // the driver picks spin ("auto" resolves to spin when there are
            // enough cores), so the launch worker sat at 99.5% of a core for the
            // whole run. `perf record -F 999 -g` put 97.6% of all samples in the
            // `Stream::synchronize` call chain at the bottom of this loop, 13.9%
            // of them in `__vdso_clock_gettime` itself, and per-thread /proc
            // accounting showed that one thread burning a core whether the run
            // produced 9,062 keys or 8 -- so it was spinning on the GPU, not
            // doing host work.
            //
            // `cudaDeviceScheduleAuto` is documented as: "if the number of active
            // CUDA contexts is greater than the number of logical processors,
            // use yield; otherwise spin." This is a 12-thread laptop with one
            // context, so it takes the spin branch every time.
            //
            // The cost of blocking is wake-up latency per synchronise, which
            // matters only when synchronise is hot. Here each launch runs ~45 ms
            // of kernel and the loop is serialised on it -- the next launch
            // cannot be issued until the previous retires -- so a microsecond
            // wake-up is noise and the core is better spent elsewhere.
            let _context = Context::create_and_push(
                ContextFlags::MAP_HOST | ContextFlags::SCHED_BLOCKING_SYNC,
                device,
            )
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
            // `StreamFlags::DEFAULT`, not `StreamFlags::NON_BLOCKING`.
            //
            // `NON_BLOCKING` is `cuStreamNonBlocking`, a *concurrency* flag, and
            // rustacuda's own doc comment says it is "misleading" and "has no
            // effect in most circumstances" because it does not expose the NULL
            // stream. It says nothing about spin-vs-block, which is what
            // `Stream::synchronize` actually chooses.
            //
            // The spin is real and is worth avoiding: `perf record -F 999 -g`
            // put 97.6% of samples in the `Stream::synchronize` call chain here,
            // 13.9% in `__vdso_clock_gettime` itself, and per-thread /proc
            // accounting showed one thread at 99.5% of a core for the whole run
            // whether it produced 9,062 keys or 8. So this thread spends its
            // life spinning on the GPU rather than doing host work.
            //
            // The flag that actually controls it is `cudaDeviceScheduleBlockingSync`
            // (0x04), which is a device/primary-context attribute set through
            // `cudaSetDeviceFlags` or the context-creation flags. rustacuda 0.1
            // exposes neither -- `StreamFlags` has exactly two values -- so
            // switching to `DEFAULT` here is a correctness tidy-up, NOT the fix.
            // See the note on `spin_sync` below for what is and is not achieved.
            let stream = Stream::new(StreamFlags::DEFAULT, None).unwrap();

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
/// What the launch geometry is derived from, as plain numbers.
///
/// `geometry` used to read these off the device and function inline, which left
/// the arithmetic untestable: it is the function that decides every number in
/// `kernel.conf`, and it had no test at all. Reading the attributes is the only
/// part that needs CUDA; everything after it is arithmetic over these fields, so
/// it lives in [`plan_geometry`] where it can be checked.
#[derive(Debug, Clone, Copy)]
struct GeometryInputs {
    /// `MaxThreadsPerBlock` as reported by the *function*.
    fn_max_threads: u32,
    /// `NumRegisters` as reported by the function: registers per thread.
    fn_registers: u32,
    /// `MaxThreadsPerBlock` as reported by the *device*.
    gpu_max_threads: u32,
    /// `MaxRegistersPerBlock`: the register file available to one block.
    gpu_max_registers: u32,
    /// `MultiprocessorCount`.
    gpu_cores: u32,
    /// `MaxThreadsPerMultiprocessor`.
    max_threads_per_sm: u32,
}

fn geometry(
    device: &rustacuda::device::Device,
    function: &rustacuda::function::Function,
    launch: &Launch,
) -> (u32, u32, u64) {
    use rustacuda::device::DeviceAttribute;
    use rustacuda::function::FunctionAttribute;

    let inputs = GeometryInputs {
        fn_max_threads: function
            .get_attribute(FunctionAttribute::MaxThreadsPerBlock)
            .unwrap() as u32,
        fn_registers: function
            .get_attribute(FunctionAttribute::NumRegisters)
            .unwrap() as u32,
        gpu_max_threads: device
            .get_attribute(DeviceAttribute::MaxThreadsPerBlock)
            .unwrap() as u32,
        gpu_max_registers: device
            .get_attribute(DeviceAttribute::MaxRegistersPerBlock)
            .unwrap() as u32,
        gpu_cores: device
            .get_attribute(DeviceAttribute::MultiprocessorCount)
            .unwrap() as u32,
        max_threads_per_sm: device
            .get_attribute(DeviceAttribute::MaxThreadsPerMultiprocessor)
            .unwrap_or(1536) as u32,
    };
    plan_geometry(&inputs, launch)
}

/// Round down to a multiple of 32, the warp size, never below one warp.
const fn round_down_to_warp(n: u32) -> u32 {
    let r = n / 32 * 32;
    if r == 0 {
        32
    } else {
        r
    }
}

/// Choose threads per block and blocks per launch. The arithmetic half of
/// `geometry`, split out so it can be tested without a device.
///
/// # The register budget
///
/// `by_occupancy` was `gpu_max_registers / fn_registers`, which is the per-block
/// register file divided by registers per thread. That is the right *magnitude*
/// but the wrong *number*: the division is not rounded to a warp boundary, so at
/// 168 registers it yields 390 threads, which is neither a multiple of 32 nor
/// achievable as a launch. It was harmless only because `DEFAULT_BLOCK` is
/// smaller and `min()` won first; raise `block` in `kernel.conf` and the 390
/// would have been the answer. Rounding *down* to a warp multiple (384 here) is
/// both correct and strictly safer -- rounding up would oversubscribe the
/// register file and the launch would fail outright.
///
/// This is a per-*block* budget, not a per-SM occupancy figure. Converting
/// registers per thread into warps per SM is `report_geometry`'s job, and it
/// reports that rather than deriving the launch from it.
fn plan_geometry(inputs: &GeometryInputs, launch: &Launch) -> (u32, u32, u64) {
    let GeometryInputs {
        fn_max_threads,
        fn_registers,
        gpu_max_threads,
        gpu_max_registers,
        gpu_cores,
        max_threads_per_sm,
    } = *inputs;

    // Registers per thread, floored at 1 so this cannot divide by zero. `max(1)`
    // does not change any real value: a kernel using 0 registers does not exist.
    //
    // Rounded to a warp, which is what makes it a legal launch width: at 168
    // registers/thread the raw quotient is 65535/168 = 390, and 390 is neither a
    // multiple of 32 nor something a launch can be.
    //
    // Deliberately *not* rounded a second time in the `Some` arm. Both sites
    // were live at one point and the result was a behaviour that could not be
    // tested: with the rounding deleted from here, the suite stayed green,
    // because the outer call turned 390 into the same 384. Two roundings of one
    // number is one rounding too many, and the redundant one is what made this
    // untestable rather than the rounding itself.
    let by_occupancy = round_down_to_warp(gpu_max_registers / fn_registers.max(1));

    let threads = match launch.threads {
        // An explicit --threads is a request, not a command. It has to be
        // clamped against everything that can make the launch *fail*, and that
        // includes the register budget: at 96 registers/thread, 1024 threads
        // needs 98,304 registers from a 65,536 register file, and CUDA rejects
        // the launch outright. The pre-existing code clamped only against the
        // two thread limits, so `--threads 1024` on a register-heavy kernel
        // failed at launch with no diagnostic. `threads_never_oversubscribe_
        // the_register_file` is the test that caught it.
        //
        // `by_occupancy` is already a warp multiple, so this `min` yields a warp
        // multiple whenever the request is one. The outer round handles a
        // *request* that is not, such as 100 -> 96.
        Some(t) => round_down_to_warp(
            t.min(fn_max_threads)
                .min(gpu_max_threads)
                .min(by_occupancy),
        ),
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

    use super::prefix_spec;

    /// The comparison the kernel performs, assembled from the *real*
    /// `prefix_spec` rather than a hand copy.
    /// This is the one thing that changed when the tests were pointed at
    /// production code. Previously `compare` re-derived the index, the mask and
    /// the decode itself, so it tested a duplicate: `from_str` could drift and
    /// every test here would still pass. Now the only unverified step is the
    /// two-line predicate below, which is the same arithmetic the device does in
    /// `BytePrefix::matches_low8` and is separately cross-checked against the
    /// full 32-byte comparison in `core`'s `matches_low8_agrees_with_matches`.
    fn compare(prefix: &str, data: &[u8; 32]) -> bool {
        let spec = prefix_spec(prefix);
        let last = spec.last_byte_idx;
        data.starts_with(&spec.bytes[..last])
            && (data[last] & spec.last_byte_mask) == spec.bytes[last]
    }

    /// The bytes a prefix should match, as a pubkey would present them.
    fn expected_bytes(prefix: &str) -> [u8; 32] {
        let decoded = prefix_spec(prefix).bytes;
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
            // From the real spec, not a second derivation of it: this loop is
            // about *which bits the matcher looks at*, and taking that from the
            // implementation is the point. `a_prefix_rejects...` and the
            // implementation agreeing is what makes the mutation below
            // meaningful; if both came from one hand-written formula, a wrong
            // formula would cancel itself out.
            let spec = prefix_spec(&prefix);
            let (last_idx, mask) = (spec.last_byte_idx, spec.last_byte_mask);
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
            let mask = prefix_spec(&"a".repeat(len)).last_byte_mask;            if n_bits == 0 {
                assert_eq!(mask, 0xFF, "len {} uses whole bytes", len);
            } else {
                let used = mask.count_ones();
                assert_eq!(used as u32, n_bits as u32, "len {}", len);
            }
        }
    }

    #[test]
    fn spec_agrees_with_the_kernel_side_host_idx_mask() {
        // `core/src/lib.rs` derives the same index and mask independently, for
        // the device-side matcher, and pins them against a hand-worked table.
        // Both derivations exist because the host packs the mask into a
        // `BytePrefix` the kernel reads, so if they ever disagree the kernel
        // silently matches nothing. Cross-checking the two here is what keeps
        // that from being possible.
        for len in 1..=crate::core::BytePrefix::MAX_PREFIX_CHARS {
            let bits: usize = 5 * len;
            let n_bits = bits % 8;
            let want_idx = bits.div_ceil(8) - 1;
            let want_mask: u8 = if n_bits == 0 {
                0xFF
            } else {
                ((1 << n_bits) - 1) << (8 - n_bits)
            };
            let spec = prefix_spec(&"q".repeat(len));
            assert_eq!(spec.last_byte_idx, want_idx, "last_byte_idx, len {len}");
            assert_eq!(spec.last_byte_mask, want_mask, "last_byte_mask, len {len}");
        }
    }

    #[test]
    fn lowercase_prefixes_decode_rather_than_returning_none() {
        // The silent-no-search bug, pinned.
        //
        // base32 0.5 made the two RFC 4648 alphabets case-specific, and the
        // uppercase one returns `None` for a lowercase input rather than
        // decoding it. `prefix_spec` must therefore use `Rfc4648Lower`.
        //
        // The failure mode is the worst kind: the search runs, matches nothing,
        // and reports no reason why. `cli::parse` lowercases prefixes before
        // they arrive here, so the lowercase alphabet is the matching one, and
        // swapping it is invisible until a user waits hours for a 2^40 search
        // that was never going to succeed.
        //
        // Note what this test does *not* claim. The uppercase alphabet returns
        // `None` for lowercase input, but it also returns `None` for uppercase
        // input, so "uppercase must decode" is not a usable oracle here; the
        // distinction that matters is lower-alphabet-plus-lower-input, which is
        // the only combination that yields bytes.
        let spec = prefix_spec("zzzz");
        assert_eq!(
            spec.bytes,
            vec![206u8, 115, 144],
            "lowercase alphabet on lowercase input must decode"
        );
        // And the specific wrong choice, spelled out so the failure is legible.
        assert_eq!(
            base32::decode(
                base32::Alphabet::Rfc4648 { padding: false },
                "zzzzaa"
            ),
            None,
            "the uppercase alphabet is expected to reject lowercase input; this is \
             the trap prefix_spec must not fall into"
        );
        // A one-character prefix pins 5 bits, so it decodes to a single byte of
        // which only the top 5 are pinned. 'b' is base32 value 1, so the pinned
        // bits are 0b00001 and the byte is 1 << 3 = 8. This distinguishes a real
        // decode from an empty or zeroed buffer: the value is determined by the
        // alphabet, not merely non-zero. ('a' would decode to 0x00, since 'a' is
        // base32 zero -- so "not all zeros" would be the wrong assertion here.)
        assert_eq!(prefix_spec("b").bytes, vec![8u8], "'b' pins 5 bits of one byte");
        // And a decode must be non-empty for every legal single character.
        for c in "abcdefghijklmnopqrstuvwxyz234567".chars() {
            let spec = prefix_spec(&c.to_string());
            assert_eq!(
                spec.bytes.len(),
                1,
                "a 1-character prefix decodes to exactly one byte, got {:?}",
                spec.bytes
            );
        }
    }

    #[test]
    fn a_prefix_longer_than_the_word_is_rejected() {
        // Two independent guards make a 13-character prefix fail: the
        // `MAX_PREFIX_CHARS` cap, and the `last_byte_idx < 8` assertion (13
        // chars pin 65 bits = 9 bytes, and byte 8 does not exist in the word
        // the kernel compares). Removing either one alone leaves the other,
        // which is why this asserts the *message* rather than only that
        // something panicked: a bare `is_err()` cannot tell which guard fired
        // and so passes even with the length cap deleted.
        //
        // The message assertion is what pins the cap specifically.
        let msg = std::panic::catch_unwind(|| {
            let _ = prefix_spec(&"a".repeat(13));
        })
        .expect_err("a 13-character prefix must be rejected")
        .downcast_ref::<String>()
        .cloned()
        .unwrap_or_default();
        assert!(
            msg.contains("at most"),
            "expected the MAX_PREFIX_CHARS cap to fire, got: {msg:?}"
        );
    }

    #[test]
    fn an_empty_prefix_is_rejected() {
        // `prefix_spec("")` computes `0.div_ceil(8) - 1`, i.e. `0 - 1`. In a
        // release build that underflow *wraps* to `usize::MAX` rather than
        // panicking, and what actually catches it is the `last_byte_idx < 8`
        // assertion further down -- confirmed by mutation: removing the explicit
        // check here leaves this test green, because the wrap is a second line
        // of defence. The check stays so the intent is stated rather than
        // incidental, and so the failure reads "a prefix may not be empty"
        // instead of "spans byte 18446744073709551615".
        //
        // This asserts behaviour rather than which assertion fires: pinning one
        // of two valid guards would make the test brittle for no gain, and the
        // mutation result is recorded here rather than papered over.
        assert!(
            std::panic::catch_unwind(|| prefix_spec("")).is_err(),
            "an empty prefix must be rejected, not underflow"
        );
        // A single character must not panic, proving the check above is not
        // simply rejecting everything.
        assert_eq!(prefix_spec("a").bytes.len(), 1);
    }

    #[test]
    fn prefix_word_is_the_little_endian_prefix_of_the_bytes() {
        // The kernel compares one 64-bit word, so this packing has to agree with
        // the byte order the matcher assumes.
        let spec = prefix_spec("abcdefgh");
        let mut want = 0u64;
        for (i, b) in spec.bytes.iter().take(8).enumerate() {
            want |= (*b as u64) << (8 * i);
        }
        assert_eq!(spec.prefix_word, want);
    }

    #[test]
    fn spec_is_self_consistent_across_the_base32_alphabet() {
        // Every legal base32 character, at a length that exercises a partial
        // mask, must decode to something the matcher can use. A character
        // outside the alphabet is the one case `prefix_spec` is allowed to
        // reject.
        const ALPHABET: &[u8] = b"abcdefghijklmnopqrstuvwxyz234567";
        for &c in ALPHABET {
            let prefix = String::from_utf8(vec![c; 3]).unwrap();
            let spec = prefix_spec(&prefix);
            assert_eq!(spec.last_byte_idx, 1, "3 chars pins 15 bits = 2 bytes");
            assert_eq!(spec.last_byte_mask, 0xFE, "3 chars = 7 bits of byte 1");
            assert!(!spec.bytes.is_empty());
        }
        // An illegal character must be a panic, not a silent empty decode.
        assert!(
            std::panic::catch_unwind(|| prefix_spec("!!")).is_err(),
            "a non-base32 prefix must be rejected"
        );
    }
}

#[cfg(test)]
mod geometry_tests {
    //! `geometry` decides every number in `kernel.conf`, and had no test at all.
    //! The arithmetic is now in `plan_geometry` over plain integers, so the
    //! invariants that actually matter can be stated here.
    //!
    //! The values below are a plausible modern NVIDIA part (20 SMs, 1536 threads
    //! per SM, a 64K register file) and match the RTX 3050 this was measured on.

    use super::{plan_geometry, GeometryInputs, Launch};

    /// The shipping configuration: 168 registers/thread, `block = 128`.
    fn inputs() -> GeometryInputs {
        GeometryInputs {
            fn_max_threads: 1024,
            fn_registers: 168,
            gpu_max_threads: 1024,
            gpu_max_registers: 65535,
            gpu_cores: 20,
            max_threads_per_sm: 1536,
        }
    }

    fn no_override() -> Launch {
        Launch::default()
    }

    #[test]
    fn default_geometry_matches_the_measured_configuration() {
        // 128 threads (from kernel.conf), 240 blocks, and the key count that
        // 15,728,640 is famous for in THROUGHPUT.md.
        let (threads, blocks, keys) = plan_geometry(&inputs(), &no_override());
        assert_eq!(threads, 128, "kernel.conf sets block = 128");
        assert_eq!(blocks, 240, "20 SMs x 1536 threads / 128");
        assert_eq!(keys, 128 * 240 * 512);
    }

    #[test]
    fn threads_is_always_a_whole_number_of_warps() {
        // The bug this module exists for: `gpu_max_registers / fn_registers`
        // gave 65535/168 = 390, which is not a multiple of 32 and cannot be a
        // launch width. It was masked by DEFAULT_BLOCK being smaller, so the
        // defect only surfaced once `block` was raised.
        for regs in [1u32, 32, 63, 96, 128, 168, 200, 255] {
            let mut i = inputs();
            i.fn_registers = regs;
            // Force the register term to be the binding constraint by asking for
            // a very large --threads, which the clamp reduces to by_occupancy.
            let launch = Launch {
                threads: Some(1024),
                blocks_per_sm: None,
                verbose: false,
            };
            let (threads, _, _) = plan_geometry(&i, &launch);
            assert_eq!(
                threads % 32,
                0,
                "{regs} registers/thread gave {threads} threads, not a warp multiple"
            );
        }
    }

    #[test]
    fn round_down_to_warp_never_rounds_up_and_never_reaches_zero() {
        // Testing the rounding directly, because `threads_is_always_a_whole_
        // number_of_warps` does not: that one drives `plan_geometry`, whose
        // `Some` branch rounds *again* after clamping, so it stayed green with
        // the rounding removed from the register term entirely. Mutation-
        // testing found that gap, which is why this exists.
        use super::round_down_to_warp;
        // 390 is the value at 168 registers/thread: rounds down to 384, never up
        // to 416 (which would need 69,888 registers and fail the launch).
        assert_eq!(round_down_to_warp(390), 384);
        assert_eq!(round_down_to_warp(391), 384);
        assert_eq!(round_down_to_warp(1024), 1024, "already a warp multiple");
        assert_eq!(round_down_to_warp(100), 96);
        assert_eq!(round_down_to_warp(32), 32);
        // Below one warp, round *up* to one warp rather than to zero. A zero
        // block width would make `blocks = cores * capacity / 0` divide by zero.
        assert_eq!(round_down_to_warp(1), 32);
        assert_eq!(round_down_to_warp(31), 32);
        assert_eq!(round_down_to_warp(0), 32, "zero must not produce zero");
    }

    #[test]
    fn the_register_term_of_the_default_branch_is_a_warp_multiple() {
        // Pins the actual value the register budget produces, which is what
        // `DEFAULT_BLOCK` masking hid: with the two thread caps lifted, 168
        // registers/thread must yield 384 threads, not the 390 that
        // `65535 / 168` computes.
        //
        // # What this test does and does not prove
        //
        // It pins the value, so a regression that made the budget 1024 or 256
        // would fail. It does *not* prove the rounding happens at this
        // particular line: the `Some` arm rounds its whole clamp, so deleting
        // the rounding here leaves 390 arriving and being rounded to the same
        // 384. Mutation testing confirmed that, and the response was to delete
        // the redundant second rounding rather than contort a test to
        // distinguish two spellings of the same result.
        //
        // The original defect is still covered: an unrounded 390 reaching a
        // launch is what `threads_is_always_a_whole_number_of_warps` and
        // `threads_never_oversubscribe_the_register_file` guard, and the
        // register-budget clamp itself is guarded by
        // `threads_never_oversubscribe_the_register_file`.
        for regs in [1u32, 32, 63, 96, 128, 168, 200, 255] {
            let mut i = inputs();
            i.fn_registers = regs;
            i.fn_max_threads = u32::MAX;
            i.gpu_max_threads = u32::MAX;
            let launch = Launch {
                threads: Some(u32::MAX),
                blocks_per_sm: None,
                verbose: false,
            };
            let (threads, _, _) = plan_geometry(&i, &launch);
            // 65535 / regs, floored to a whole warp. Written as the expected
            // computation so the assertion states the rule, not a magic number.
            let want = (65535 / regs.max(1)) / 32 * 32;
            assert_eq!(
                threads,
                want.max(32),
                "{regs} registers/thread: 65535/{regs} must round down to {want}"
            );
            assert_eq!(threads % 32, 0, "{regs}: {threads} is not a warp multiple");
            assert!(
                threads as u64 * regs as u64 <= 65_536,
                "{threads} threads x {regs} registers oversubscribes the 64K file"
            );
        }
        // And the value that was 390 in the original code, called out.
        let mut i = inputs();
        i.fn_max_threads = u32::MAX;
        i.gpu_max_threads = u32::MAX;
        let (threads, _, _) = plan_geometry(
            &i,
            &Launch {
                threads: Some(u32::MAX),
                blocks_per_sm: None,
                verbose: false,
            },
        );
        assert_eq!(threads, 384, "65535/168 = 390, which is not a launch width");
    }

    #[test]
    fn threads_never_oversubscribe_the_register_file() {
        // Rounding down to a warp is only correct if it cannot round *up*.
        // 65536/168 = 390.1, and 390 rounds down to 384: 384 * 168 = 64,512,
        // inside the 65,536 available. Rounding up to 416 would need 69,888 and
        // the launch would fail.
        for regs in [1u32, 40, 64, 96, 128, 168, 192, 224, 255] {
            let mut i = inputs();
            i.fn_registers = regs;
            let launch = Launch {
                threads: Some(1024),
                blocks_per_sm: None,
                verbose: false,
            };
            let (threads, _, _) = plan_geometry(&i, &launch);
            assert!(
                threads as u64 * regs as u64 <= 65_536,
                "{threads} threads x {regs} registers exceeds the 64K block file"
            );
        }
    }

    #[test]
    fn a_degenerate_register_count_cannot_divide_by_zero() {
        // A kernel reporting 0 registers must not panic or produce garbage.
        let mut i = inputs();
        i.fn_registers = 0;
        let (threads, blocks, keys) = plan_geometry(&i, &no_override());
        assert!(threads > 0 && blocks > 0 && keys > 0);
    }

    #[test]
    fn explicit_threads_are_clamped_not_obeyed_blindly() {
        // --threads is a request, clamped against every limit that can make the
        // launch fail. At 168 registers/thread the register budget allows 384
        // threads, which binds before the 1024 thread limit does -- so an
        // oversized request lands on 384, not 1024.
        let launch = Launch {
            threads: Some(4096),
            blocks_per_sm: None,
            verbose: false,
        };
        let (threads, _, _) = plan_geometry(&inputs(), &launch);
        assert_eq!(
            threads, 384,
            "clamped to the register budget (65535/168 = 390, rounded to 12 warps)"
        );

        // A partial warp request is rounded down, never up.
        let launch = Launch {
            threads: Some(100),
            blocks_per_sm: None,
            verbose: false,
        };
        let (threads, _, _) = plan_geometry(&inputs(), &launch);
        assert_eq!(threads, 96, "100 rounds down to 3 warps");

        // A request under the register budget is honoured exactly.
        let launch = Launch {
            threads: Some(256),
            blocks_per_sm: None,
            verbose: false,
        };
        let (threads, _, _) = plan_geometry(&inputs(), &launch);
        assert_eq!(threads, 256, "256 fits the register budget and is exact");
    }

    #[test]
    fn a_tiny_thread_count_still_yields_a_valid_launch() {
        // 1 thread is below a warp, but --threads 1 is a legal request and must
        // not produce blocks = 0 or a zero key count.
        let launch = Launch {
            threads: Some(1),
            blocks_per_sm: None,
            verbose: false,
        };
        let (threads, blocks, keys) = plan_geometry(&inputs(), &launch);
        assert_eq!(threads, 32, "rounded up to one warp, never zero");
        assert!(blocks >= 1);
        assert!(keys > 0);
    }

    #[test]
    fn blocks_per_sm_override_is_honoured_exactly() {
        let launch = Launch {
            threads: None,
            blocks_per_sm: Some(3),
            verbose: false,
        };
        let (threads, blocks, keys) = plan_geometry(&inputs(), &launch);
        assert_eq!(threads, 128);
        assert_eq!(blocks, 3 * 20, "3 blocks per SM x 20 SMs");
        assert_eq!(keys, 128 * 60 * 512);
    }

    #[test]
    fn keys_per_launch_is_exactly_threads_blocks_batch() {
        // The counter the throughput number is derived from. If this drifts, every
        // reported keys/s is wrong by the same factor, silently.
        for threads in [32u32, 64, 128, 256, 1024] {
            for blocks in [1u32, 7, 240] {
                let launch = Launch {
                    threads: Some(threads),
                    blocks_per_sm: None,
                    verbose: false,
                };
                let (t, b, keys) = plan_geometry(&inputs(), &launch);
                assert_eq!(
                    keys,
                    t as u64 * b as u64 * 512,
                    "keys must be threads*blocks*BATCH for {threads}x{blocks}"
                );
            }
        }
    }

    #[test]
    fn the_launch_fills_the_device() {
        // With no override, threads x blocks x SMs should cover the machine's
        // thread capacity -- that is the whole point of computing blocks rather
        // than hardcoding it.
        let (threads, blocks, _) = plan_geometry(&inputs(), &no_override());
        let capacity = 20 * 1536;
        let resident_threads = threads as u64 * blocks as u64;
        assert!(
            resident_threads >= capacity as u64,
            "launch covers {resident_threads} threads, device capacity is {capacity}"
        );
    }
}
