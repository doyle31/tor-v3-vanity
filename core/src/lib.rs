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

/// Threads per block. Each thread runs an independent batch of candidates.
pub const BLOCK: usize = 256;

/// Candidates examined per chunk. A chunk is batch-inverted together, so this
/// is also the inversion amortisation factor.
pub const CHUNK: usize = 8;

/// Candidates examined per thread per launch. Must be a multiple of `CHUNK`
/// so that every chunk is full.
pub const BATCH: usize = 64;

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

#[derive(DeviceCopy, Clone)]
#[repr(C)]
pub struct BytePrefix {
    pub byte_prefix: DevicePointer<u8>,
    pub byte_prefix_len: usize,
    pub last_byte_idx: usize,
    pub last_byte_mask: u8,
    pub out: DevicePointer<u8>,
    pub success: DevicePointer<bool>,
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
#[cfg(target_arch = "nvptx64")]
pub const TABLE_WORDS: usize = group::TABLE_LEN * POINT_WORDS;
