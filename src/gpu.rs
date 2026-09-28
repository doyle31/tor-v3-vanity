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

use failure::Error;
use tor_v3_vanity_core as core;

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
/// An empty `prefixes` is the benchmark case: the kernel then skips matching
/// entirely. When `stop` is set the loops exit after the current launch, which
/// is how a benchmark shuts the GPU down without leaving it running.
pub fn start(
    prefixes: &[String],
    sender: crossbeam_channel::Sender<[u8; 64]>,
    tries_sender: crossbeam_channel::Sender<u64>,
    stop: Option<Arc<AtomicBool>>,
    launch: Launch,
) -> Result<Dropped, Error> {
    use rustacuda::launch;
    use rustacuda::memory::DeviceBox;
    use rustacuda::prelude::*;
    use std::ffi::CString;

    rustacuda::init(CudaFlags::empty())?;
    let dropped_keys: Dropped = Arc::new(AtomicU64::new(0));
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

            let module_data = CString::new(include_str!(env!("KERNEL_PTX_PATH"))).unwrap();
            let kernel = Module::load_from_string(&module_data).unwrap();
            let function = kernel
                .get_function(std::ffi::CStr::from_bytes_with_nul(b"render\0").unwrap())
                .unwrap();

            let mut table = upload_power_table().unwrap();
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

            let matching = real_len > 0;
            let (threads, blocks, keys_per_launch) =
                geometry(&device, &function, &launch);
            if launch.verbose {
                report_geometry(&device, &function, i, threads, blocks, keys_per_launch, matching);
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
                    // captured prefix is used. At SLOTS=2048 that is 128 KiB per
                    // launch against a ~34 ms launch: ~0.04% of the budget.
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
        });
        i += 1;
    }
    if i == 0 {
        eprintln!("No cuda devices available.");
        std::process::exit(2);
    }
    Ok(dropped_keys)
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
            gpu_max_threads.min(core::BLOCK as u32),
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
