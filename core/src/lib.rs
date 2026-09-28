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
    /// One `u64` per thread, written only by the no-prefix (`--bench`) path.
    ///
    /// This is a guard, not a fix. The no-prefix path walks a `BatchState` that
    /// nothing reads afterwards, which is exactly the shape dead-code
    /// elimination targets: the whole point-addition chain is one store away
    /// from being deleted, and `--bench` would then report a "key generation"
    /// rate for a kernel doing none. Writing the final walking point here makes
    /// the chain observable through a volatile store.
    ///
    /// At the time this was added the walk was **not** actually being deleted:
    /// adding the store left the compiled kernel at 23,128 instructions, byte
    /// for byte the same instruction count as without it, and `--bench` still
    /// reported 511M keys/s. So the benchmark was already measuring the walk.
    /// The guard is kept because the exposure is real and the cost is one 8-byte
    /// store per thread, but it is not what explains the throughput numbers --
    /// see the note in `kernel.conf`.
    pub bench_sink: DevicePointer<u64>,
}

/// Expanded keys captured per prefix per launch.
///
/// Each prefix used to have a single 64-byte output slot plus a `success`
/// flag, so a launch could return at most **one** key per prefix no matter how
/// many it found. The host then read that slot once per launch and reset it.
///
/// A ring fixes it. The size is set by the *matches per launch*, which is
/// `keys_per_launch / 2^(5 * chars)`. At the shipping geometry -- 128 threads
/// x 240 blocks x 512 batch = **15,728,640 keys per launch** -- that is:
///
/// | chars | matches per launch | fits in 16384? |
/// |---|---|---|
/// | 1 | 491,520 | no |
/// | 2 | 15,360 | yes |
/// | 3 | 480 | yes |
/// | 4 | 15 | yes |
///
/// A 1-character search matches 1 key in 32, so no affordable ring holds a
/// launch's worth; that case is reported by the `dropped_keys` warning rather
/// than silently truncated. Two characters is the shortest prefix worth
/// enumerating, and 16384 holds it with headroom.
///
/// This constant has to be re-checked whenever `keys_per_launch` moves. It was
/// last sized against 1,310,720 keys per launch, when 2048 slots covered a
/// 2-character launch. Raising `batch`/`chunk` took keys per launch up 12x and
/// the ring was not revisited, which silently dropped 86.7% of 2-character
/// matches (99.5% at 1 character) until it was caught. Prefer sizing this
/// *from* the geometry over reusing a previous value.
///
/// Cost of the size: `SLOTS * SLOT_BYTES` per prefix -- 1 MiB at 16384, against
/// 128 KiB at 2048 -- and the whole ring is read back per launch because
/// `copy_to` requires equal lengths. At the measured ~14.5 launches/s that is
/// ~15 MB/s, which is ~0.1% of a PCIe 3.0 x16 link.
pub const SLOTS: usize = 16384;

/// Bytes of expanded key per slot.
pub const SLOT_BYTES: usize = 64;

#[derive(DeviceCopy, Clone)]
#[repr(C)]
pub struct BytePrefix {
    pub byte_prefix: DevicePointer<u8>,
    pub byte_prefix_len: usize,
    pub last_byte_idx: usize,
    pub last_byte_mask: u8,
    /// The first eight prefix bytes, packed little-endian.
    ///
    /// This is what the kernel actually tests: the low eight bytes of the
    /// encoding are one 64-bit word, so a prefix costs a masked compare instead
    /// of a `memcmp` against a byte array in local memory. The host fills it in
    /// from `byte_prefix`; only bytes `0..=last_byte_idx` are ever read, so the
    /// bytes past the prefix are zero and the rest of `byte_prefix` is
    /// redundant on the device.
    pub prefix_word: u64,
    /// `SLOTS * SLOT_BYTES` bytes: a ring of captured expanded keys.
    pub out: DevicePointer<u8>,
    /// Atomic count of matches published this launch, also the ring cursor.
    pub count: DevicePointer<u32>,
}

impl BytePrefix {
    /// The longest prefix the kernel can test, in base32 characters.
    ///
    /// `L` characters cover `5L` bits, and the host rounds that up to a whole
    /// number of bytes, so the highest byte it ever compares is
    /// `(5L + 7) / 8 - 1`. The kernel tests the low *eight* bytes, which
    /// `L == 12` (60 bits, byte 7) exactly fills. Longer is not a search: a
    /// 12-character prefix is a 2^60-candidate search, so the host rejects
    /// anything longer rather than letting it silently never match.
    pub const MAX_PREFIX_CHARS: usize = 12;

    pub fn matches(&self, data: &[u8]) -> bool {
        let slice =
            unsafe { core::slice::from_raw_parts(self.byte_prefix.as_raw(), self.byte_prefix_len) };
        data.starts_with(&slice[..self.last_byte_idx])
            && data[self.last_byte_idx] & self.last_byte_mask == slice[self.last_byte_idx]
    }

    /// The same predicate as [`matches`](Self::matches), over just the low eight
    /// bytes of the encoding packed little-endian (see
    /// `field::fe_to_bytes_low8`).
    ///
    /// Byte `i` of the encoding is `(w >> (8 * i)) & 0xFF`, so the test splits
    /// into a compare of everything below `last_byte_idx` and a masked compare
    /// of the byte at it. That is `matches` exactly: bytes `0..last_byte_idx`
    /// must match in full, and `data[last_byte_idx] & last_byte_mask` must equal
    /// the prefix's byte there. The mask is applied to the data side only, as in
    /// `matches`, so a prefix byte with junk in the bits the mask clears behaves
    /// exactly as it did before rather than being quietly accepted.
    ///
    /// Returns false for a prefix wider than eight bytes instead of reading past
    /// the word; the host rejects those (see [`MAX_PREFIX_CHARS`](Self::MAX_PREFIX_CHARS)).
    #[inline(always)]
    pub fn matches_low8(&self, w: u64) -> bool {
        let idx = self.last_byte_idx;
        if idx >= 8 {
            return false;
        }
        // Bytes 0..idx, and idx itself. `idx < 8`, so the shift is under 64.
        let low = (1u64 << (8 * idx)) - 1;
        if (w & low) != (self.prefix_word & low) {
            return false;
        }
        let mask = self.last_byte_mask as u64;
        (w >> (8 * idx)) & mask == (self.prefix_word >> (8 * idx)) & 0xFF
    }

    /// Pack the first eight bytes of `prefix` little-endian into a
    /// `prefix_word`.
    ///
    /// Host side. Bytes past the end of `prefix` are zero, which is what
    /// `matches_low8` needs: it only ever reads bytes `0..=last_byte_idx`.
    pub fn pack_prefix_word(prefix: &[u8]) -> u64 {
        let mut w = 0u64;
        for (i, b) in prefix.iter().take(8).enumerate() {
            w |= (*b as u64) << (8 * i);
        }
        w
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::{fe_to_bytes, fe_to_bytes_low8, Fe};
    #[allow(unused_imports)]
    use std::vec::Vec;

    /// A deterministic xorshift, so a failure replays.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
        fn fe(&mut self) -> Fe {
            let mask = (1u64 << 51) - 1;
            [
                self.next() & mask,
                self.next() & mask,
                self.next() & mask,
                self.next() & mask,
                self.next() & mask,
            ]
        }
        fn bytes(&mut self, n: usize) -> Vec<u8> {
            (0..n).map(|_| self.next() as u8).collect()
        }
    }

    /// The `last_byte_idx` / `last_byte_mask` the host derives from a prefix of
    /// `l` base32 characters, copied from `BytePrefixOwned::from_str`.
    ///
    /// A prefix pins `5 * l` bits, so it spans `ceil(5 * l / 8)` bytes and the
    /// last of them is partial unless `5 * l` is a whole number of bytes. The
    /// host derives that as `bits.div_ceil(8) - 1` with a full `0xFF` mask in the
    /// whole-byte case, and these tests are the device crate's only record of
    /// what the host passes down, so they have to track it exactly.
    ///
    /// This used to reproduce the older host arithmetic, `5 * l / 8 - 1`
    /// followed by an increment whenever the mask was nonzero. That had two
    /// faults, and this helper faithfully carried both:
    ///
    /// * at `l == 1` the subtraction underflowed, so the increment panicked in a
    ///   debug build -- these four tests could not run at all;
    /// * a full-width mask is exactly the whole-byte case, so making that mask
    ///   `0xFF` (the fix for 8-character prefixes matching nothing) also tripped
    ///   the increment and pushed the index one byte past the prefix.
    ///
    /// `div_ceil` has neither wraparound nor a special case to get backwards.
    fn host_idx_mask(l: usize) -> (usize, u8) {
        let bits = 5 * l;
        let n_bits = bits % 8;
        let last_byte_idx = bits.div_ceil(8) - 1;
        let last_byte_mask = if n_bits == 0 {
            0xFF
        } else {
            ((1u8 << n_bits) - 1) << (8 - n_bits)
        };
        (last_byte_idx, last_byte_mask)
    }

    /// A `BytePrefix` over a host-resident prefix buffer, plus the buffer itself.
    ///
    /// `DevicePointer` is `#[repr(transparent)]` over a raw pointer and both
    /// matchers only read through it, so one pointing at host memory exercises
    /// exactly the code the device runs, with no device involved.
    fn prefix(bytes: Vec<u8>, l: usize) -> (BytePrefix, Vec<u8>) {
        let (last_byte_idx, last_byte_mask) = host_idx_mask(l);
        let bp = BytePrefix {
            byte_prefix: unsafe { DevicePointer::wrap(bytes.as_ptr() as *mut u8) },
            byte_prefix_len: bytes.len(),
            last_byte_idx,
            last_byte_mask,
            prefix_word: BytePrefix::pack_prefix_word(&bytes),
            out: DevicePointer::null(),
            count: DevicePointer::null(),
        };
        (bp, bytes)
    }

    #[test]
    fn host_idx_mask_matches_the_host_derivation() {
        // Pinned against the table worked out by hand, because every
        // equivalence test below is only as good as this derivation.
        let want = [
            (0usize, 0xF8u8), // 1 char,  5 bits:  byte 0, high 5
            (1, 0xC0),        // 2 chars, 10 bits: byte 0, byte 1 high 2
            (1, 0xFE),        // 3 chars, 15 bits: byte 0, byte 1 high 7
            (2, 0xF0),        // 4 chars, 20 bits: bytes 0-1, byte 2 high 4
            (3, 0x80),        // 5 chars, 25 bits: bytes 0-2, byte 3 high 1
            (3, 0xFC),        // 6 chars, 30 bits: bytes 0-2, byte 3 high 6
            (4, 0xE0),        // 7 chars, 35 bits: bytes 0-3, byte 4 high 3
            (4, 0xFF),        // 8 chars, 40 bits: bytes 0-4 in full
            (5, 0xF8),        // 9 chars, 45 bits: bytes 0-4, byte 5 high 5
            (6, 0xC0),        // 10 chars, 50 bits: bytes 0-5, byte 6 high 2
            (6, 0xFE),        // 11 chars, 55 bits: bytes 0-5, byte 6 high 7
            (7, 0xF0),        // 12 chars, 60 bits: bytes 0-6, byte 7 high 4
        ];
        for (l, w) in want.iter().enumerate() {
            let l = l + 1;
            assert_eq!(host_idx_mask(l), *w, "{} characters", l);
            assert!((w.0 as usize) < 8, "{} characters must fit the word", l);
        }
        // One past the limit is what the host must reject.
        assert!(host_idx_mask(13).0 >= 8);
    }

    #[test]
    fn matches_low8_agrees_with_matches() {
        // The property the whole y-only path rests on: testing the low eight
        // bytes of y in one word is the same predicate as testing the full
        // 32-byte compressed key byte by byte, for every prefix length the
        // host can produce.
        let mut rng = Rng(0x243F_6A88_85A3_08D3);
        for l in 1..=BytePrefix::MAX_PREFIX_CHARS {
            // Enough bytes for the comparison plus the two the host's "aa"
            // padding adds.
            for _ in 0..200 {
                let (bp, _buf) = prefix(rng.bytes(5 * l / 8 + 2), l);
                for _ in 0..8 {
                    let y = rng.fe();
                    let mut pk = fe_to_bytes(&y);
                    let low = fe_to_bytes_low8(&y);
                    assert_eq!(
                        bp.matches_low8(low),
                        bp.matches(&pk),
                        "l={} y={:?}",
                        l,
                        y
                    );
                    // The sign bit of x lives in byte 31. Setting it must make
                    // no difference for any prefix length: that is the whole
                    // reason the kernel can skip X/Z.
                    pk[31] ^= 0x80;
                    assert_eq!(bp.matches_low8(low), bp.matches(&pk), "sign bit, l={}", l);
                }
            }
        }
    }

    #[test]
    fn matches_low8_is_not_vacuous() {
        // A matcher that never fires would pass the equivalence test above for
        // the wrong reason, so pin both directions on real public keys.
        let mut base: crate::group::BaseTables = unsafe { core::mem::zeroed() };
        crate::group::build_comb_table(&mut base.comb, &mut base.eight);

        let mut scalar = [0u8; 32];
        scalar.copy_from_slice(&crate::sha512::sha512(&[7u8; 32])[..32]);
        scalar[0] &= 248;
        scalar[31] &= 63;
        scalar[31] |= 64;

        // Two distinct keys: the seed's, and the next one in the walk.
        let mut next = scalar;
        crate::kernel::add_clamped_scalar(&mut next, 8);

        let mut keys: Vec<[u8; 32]> = Vec::new();
        let mut lows: Vec<u64> = Vec::new();
        for s in [scalar, next].iter() {
            let p = crate::group::scalarmult_base_comb(s, &base.comb);
            let y = crate::field::fe_mul(&p.y, &crate::field::fe_invert(&p.z));
            keys.push(crate::group::ge_to_bytes_with_zi(&p, &crate::field::fe_invert(&p.z)));
            lows.push(fe_to_bytes_low8(&y));
        }
        assert_ne!(keys[0], keys[1], "the two walk steps must be different keys");

        for l in 1..=BytePrefix::MAX_PREFIX_CHARS {
            // The host's prefix bytes: the key's first bytes with the bits the
            // partial mask clears zeroed, which is exactly what the "aa" the
            // host appends before base32-decoding does.
            let (idx, mask) = host_idx_mask(l);
            let mut bytes = keys[0][..idx + 1].to_vec();
            bytes[idx] &= mask;
            let (bp, _buf) = prefix(bytes, l);

            assert!(bp.matches(&keys[0]), "full key, l={}", l);
            assert!(bp.matches_low8(lows[0]), "low word, l={}", l);
            assert!(!bp.matches(&keys[1]), "other key must not match, l={}", l);
            assert!(!bp.matches_low8(lows[1]), "other key, low word, l={}", l);
        }
    }

    #[test]
    fn matches_low8_rejects_a_prefix_wider_than_the_word() {
        // Unreachable from the host, which rejects prefixes that need more than
        // eight bytes. It must fail closed rather than shift by 64 or match a
        // truncated prefix.
        let (mut bp, _buf) = prefix(vec![0xAAu8; 12], BytePrefix::MAX_PREFIX_CHARS);
        assert!(!bp.matches_low8(u64::MAX));
        bp.last_byte_idx = 31;
        assert!(!bp.matches_low8(u64::MAX));
        bp.last_byte_idx = 8;
        assert!(!bp.matches_low8(u64::MAX));
    }

    #[test]
    fn prefix_word_is_little_endian() {
        let w = BytePrefix::pack_prefix_word(&[1, 2, 3, 4, 5, 6, 7, 8, 9]);
        assert_eq!(w, 0x0807_0605_0403_0201);
        assert_eq!(BytePrefix::pack_prefix_word(&[0xFF]), 0xFF);
        assert_eq!(BytePrefix::pack_prefix_word(&[]), 0);
    }
}

