//! Device-side ed25519 for the tor-v3 vanity search.
//!
//! The host and device share the parameter structs below, so they stay here
//! rather than in the kernel module. The crypto lives in three focused
//! modules: `field` (Curve25519 arithmetic), `group` (Edwards group
//! operations and the batched walk), and `sha512`.

#![no_std]
#![cfg_attr(target_arch = "nvptx64", feature(abi_ptx, stdarch_nvptx))]

// The device crate is `no_std`, but the unit tests need `Vec`, `format!` and
// friends. Pulling in `std` under `cfg(test)` lets `cargo test` exercise the
// exact same sources the GPU build compiles, which is how the crypto is
// validated against the RFC 8032 vectors.
#[cfg(test)]
#[macro_use]
extern crate std;

pub mod field;
pub mod group;
pub mod sha512;

use rustacuda_core::DevicePointer;
use rustacuda_derive::DeviceCopy;

// Compile-time tunables.
//
// These size local arrays (`zs`, `scratch` in the kernel) and drive loop
// unrolling, so they must be constants: a runtime value would put the chunk
// buffers in local memory and cost far more than it saves. They are therefore
// overridable at *build* time via environment variables, which is what makes
// sweeping them practical -- `T3V_CHUNK=4 cargo build --release`.
//
// The defaults below are the measured optimum for this kernel on an RTX 3050
// laptop, and the reason is not obvious. A chunk is batch-inverted together,
// so the inversion cost per candidate is 1/CHUNK, and a field inversion costs
// about as much as 30 field multiplies -- more than the point arithmetic it
// serves. So CHUNK wants to be as large as possible. BATCH amortises the
// per-batch fixed cost (one SHA-512 and one scalar multiplication) over
// CHUNK-sized chunks; the comb made the scalar multiplication ~3x cheaper,
// which moved the optimum hard: BATCH=64/CHUNK=8 measured 62.9M keys/s and
// BATCH=512/CHUNK=256 measures 138.9M.
//
// Going past this is not free. At CHUNK=256 the chunk buffers are 20 KiB per
// thread of local memory, though registers *drop* to 168 (from the 255 ceiling)
// because BATCH == CHUNK collapses the loop to a single chunk -- 12 warps per SM
// instead of 8. CHUNK=512 ties on speed for double the memory, and BATCH=1024
// is no better, so 512/256 is the knee.

/// Threads per block. Only an upper bound: the host picks the real value from
/// the kernel's register usage (see `--threads`).
pub const BLOCK: usize = match option_env!("T3V_BLOCK") {
    Some(s) => parse_usize(s.as_bytes()),
    None => 256,
};

/// Candidates examined per chunk. A chunk is batch-inverted together, so this
/// is also the inversion amortisation factor.
pub const CHUNK: usize = match option_env!("T3V_CHUNK") {
    Some(s) => parse_usize(s.as_bytes()),
    None => 256,
};

/// Candidates examined per thread per launch. Must be a multiple of `CHUNK`
/// so that every chunk is full.
pub const BATCH: usize = match option_env!("T3V_BATCH") {
    Some(s) => parse_usize(s.as_bytes()),
    None => 512,
};

/// Minimal compile-time `usize` parser: only leading decimal digits, no sign or
/// whitespace, so a typo in an env var yields 0 (and trips the assertions in
/// `render`) rather than a silent nonsense value.
pub(crate) const fn parse_usize(bytes: &[u8]) -> usize {
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

// The batching logic compiles for the host too, so it can be unit tested
// against RFC 8032 vectors; only the entry point needs the GPU intrinsics.
pub mod kernel;

#[derive(DeviceCopy, Clone)]
#[repr(C)]
pub struct KernelParams {
    pub seed: DevicePointer<u8>,
    pub byte_prefixes: DevicePointer<BytePrefix>,
    pub byte_prefixes_len: usize,
}

/// Expanded keys captured per prefix per launch.
///
/// Each prefix used to have a single 64-byte output slot plus a `success`
/// flag, so a launch could return at most **one** key per prefix no matter how
/// many it found. The host then read that slot once per launch and reset it.
/// With 1,310,720 keys per launch, a 2-character prefix matches ~1,280 times
/// per launch and a 3-character prefix ~40, but only one survived.
///
/// Measured effect: the GPU ran at 39M keys/s while keys actually arrived at
/// 24/s -- one per 1.6 million tried, where the expected rate is one per 1,024.
///
/// A ring fixes it, and the readback is proportional to the keys *actually
/// captured*, not to the ring size, so a generous ring costs one allocation and
/// no extra transfer. 2048 slots covers the matches per launch of a 2-character
/// prefix (~1,280) with headroom, after which the only limit is the GPU itself.
/// Prefixes of 3 characters or more (~40 matches per launch) never fill it.
pub const SLOTS: usize = 2048;

/// Bytes of expanded key per slot.
pub const SLOT_BYTES: usize = 64;

#[derive(DeviceCopy, Clone)]
#[repr(C)]
pub struct BytePrefix {
    pub byte_prefix: DevicePointer<u8>,
    pub byte_prefix_len: usize,
    pub last_byte_idx: usize,
    pub last_byte_mask: u8,
    /// `SLOTS * SLOT_BYTES` bytes: a ring of captured expanded keys.
    pub out: DevicePointer<u8>,
    /// Atomic count of matches published this launch, also the ring cursor.
    pub count: DevicePointer<u32>,
}

impl BytePrefix {
    pub fn matches(&self, data: &[u8]) -> bool {
        let slice =
            unsafe { core::slice::from_raw_parts(self.byte_prefix.as_raw(), self.byte_prefix_len) };
        data.starts_with(&slice[..self.last_byte_idx])
            && data[self.last_byte_idx] & self.last_byte_mask == slice[self.last_byte_idx]
    }
}

/// A `no_std` device crate needs a panic handler. Trap on the GPU, which
/// surfaces as a kernel launch failure on the host.
#[cfg(target_arch = "nvptx64")]
#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    use core::arch::nvptx::*;
    unsafe { trap() }
}

/// One entry of the precomputed `[2^i]B` ladder, as the host sees it.
///
/// The table is built on the host (which has an allocator) and uploaded once,
/// then shared by every thread. A field element is five 51-bit limbs in u64
/// words and a point is four of them in extended coordinates, so an entry is
/// exactly 20 u64s with no padding.
#[cfg(target_arch = "nvptx64")]
pub const POINT_WORDS: usize = 20;

/// The whole ladder, as flat u64 words: `TABLE_LEN * POINT_WORDS`.
///
/// Retained for the power-table reference path; the kernel itself now uploads
/// `group::BASE_TABLE_WORDS` instead (see `group::BaseTables`).
#[cfg(target_arch = "nvptx64")]
pub const TABLE_WORDS: usize = group::TABLE_LEN * POINT_WORDS;
