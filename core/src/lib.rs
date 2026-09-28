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
    // --- the 64-bit view, precomputed on the host -----------------------
    //
    // `matches` is the reference: it compares bytes of the full 32-byte
    // encoding, of which a prefix sees at most eight. `matches_word` reaches
    // the same verdict from one `u64`, and these four fields are that verdict's
    // inputs, so the kernel does no prefix-byte loading and no per-candidate
    // shift at all. See `prefix_word_masks` for how they are derived and
    // `matches_word_is_equivalent_to_matches` for the proof-by-test.
    pub full_mask: u64,
    pub full_val: u64,
    pub top_mask: u64,
    pub top_val: u64,
}

/// Pack the decoded prefix bytes into the four `u64`s `matches_word` compares
/// against: `(full_mask, full_val, top_mask, top_val)`.
///
/// `matches` accepts a candidate when
///
/// ```text
///     data[0 .. j] == slice[0 .. j]     &&     data[j] & m == slice[j]
/// ```
///
/// for `j = last_byte_idx` and `m = last_byte_mask`. `data` is the
/// little-endian encoding of `y`, so `data[i]` is bits `8i..8i+8` of `y` and the
/// two conditions are exactly
///
/// ```text
///     (word & (2^(8j) - 1))          == LE_pack(slice[0 .. j])
///     (word & (m << 8j))             == (slice[j] << 8j)
/// ```
///
/// -- a masked 64-bit compare each, with the masks and values pre-shifted into
/// place. Two details are load-bearing:
///
/// * The right-hand side of the second compare is the **whole** prefix byte, not
///   the masked byte. `matches` really does compare `data[j] & m` against the
///   unmasked `slice[j]`, so a nonzero bit of `slice[j]` outside the mask makes
///   it reject. Shifting `slice[j]` up by `8j` rather than masking it preserves
///   that, and it is what keeps this equivalent for *any* bytes, not just the
///   ones a base32 prefix can produce.
/// * `m == 0` happens when the prefix is a whole number of bytes (8 base32
///   characters, `5L % 8 == 0`). Then `matches` degenerates to
///   `0 == slice[j]`, i.e. it accepts only prefixes whose last byte is zero --
///   a pre-existing bug in the host's mask arithmetic, faithfully reproduced
///   here rather than silently "fixed", so the two paths cannot disagree about
///   which keys a search finds. The host filters every captured key against the
///   real address anyway (`main.rs`), so this costs matches, not correctness.
///
/// # Panics
///
/// If `last_byte_idx > 7`: the observable bits would run past bit 63, i.e. a
/// prefix longer than 12 base32 characters. `1u64 << 64` would be a shift by the
/// full width, so there is no correct `u64` to return; the host rejects such a
/// prefix before it ever reaches the GPU.
pub fn prefix_word_masks(bytes: &[u8], last_byte_idx: usize, last_byte_mask: u8) -> (u64, u64, u64, u64) {
    assert!(
        last_byte_idx <= 7,
        "a prefix spanning more than 8 bytes (last_byte_idx = {}) does not fit the 64-bit \
         fast path; t3v supports prefixes of up to 12 base32 characters",
        last_byte_idx
    );
    assert!(
        bytes.len() > last_byte_idx,
        "prefix of {} bytes is too short for last_byte_idx {}",
        bytes.len(),
        last_byte_idx
    );
    let k = 8 * last_byte_idx;
    // `k` is 0 for a one-character prefix, where there are no whole bytes; the
    // `if` keeps the shift out of the undefined `1 << 64` case.
    let full_mask = if k == 0 { 0 } else { (1u64 << k) - 1 };
    let mut full_val = 0u64;
    for (i, b) in bytes.iter().take(last_byte_idx).enumerate() {
        full_val |= (*b as u64) << (8 * i);
    }
    (
        full_mask,
        full_val,
        (last_byte_mask as u64) << k,
        (bytes[last_byte_idx] as u64) << k,
    )
}

impl BytePrefix {
    /// Reference matching: compare bytes of the full 32-byte compressed key.
    ///
    /// Kept as the specification the fast path is tested against, and as the
    /// only path that can serve a prefix of more than 12 characters.
    pub fn matches(&self, data: &[u8]) -> bool {
        let slice =
            unsafe { core::slice::from_raw_parts(self.byte_prefix.as_raw(), self.byte_prefix_len) };
        data.starts_with(&slice[..self.last_byte_idx])
            && data[self.last_byte_idx] & self.last_byte_mask == slice[self.last_byte_idx]
    }

    /// The same verdict as [`BytePrefix::matches`], from one 64-bit word.
    ///
    /// `word` is the low 64 bits of the compressed key, i.e.
    /// `fe_prefix_word(&fe_mul(&p.y, zi))`; see `group::ge_prefix_word`. Requires
    /// `last_byte_idx <= 7`, which `prefix_word_masks` enforces where the fields
    /// are filled in.
    ///
    /// # Where the early exit is
    ///
    /// The two compares are independent and `&&` skips the second when the
    /// first fails, so a candidate is rejected on the whole-byte half of the
    /// prefix without touching the partial byte. That is a *per lane* saving
    /// only: a warp skips the second compare only when no lane in it has passed
    /// the first, and with one prefix of `n` characters requested that is
    /// `(1 - 2^-n)^32`, i.e. `2^(32n)` in the usual case. In practice both
    /// compares execute every time, so nothing cleverer is built on top of them.
    #[inline(always)]
    pub fn matches_word(&self, word: u64) -> bool {
        (word & self.full_mask) == self.full_val && (word & self.top_mask) == self.top_val
    }
}

/// How many prefix bytes `base32::decode(s + "aa")` produces for a prefix of
/// `chars` base32 characters: the crate truncates rather than rounds up, and
/// both matching methods must stay inside that many bytes.
///
/// `#[cfg(test)]` because nothing on the device needs it -- the host derives the
/// byte count from the decode itself -- but the equivalence tests need to build
/// prefixes the same way the host does.
#[cfg(test)]
pub(crate) fn prefix_byte_len(chars: usize) -> usize {
    (5 * (chars + 2)) / 8
}

/// The `last_byte_idx` / `last_byte_mask` the host computes for a prefix of
/// `chars` base32 characters, copied from `gpu::BytePrefixOwned::from_str`.
///
/// The `wrapping_sub` is the shipped behaviour, not a slip: for a
/// one-character prefix `5 * chars / 8` is 0 and the `- 1` underflows. The host
/// is built in release, where that wraps to `usize::MAX` and the `mask > 0`
/// branch immediately increments it back to 0 -- which is the right answer,
/// because a one-character prefix has no whole bytes before its partial one.
/// Spelling it with `wrapping_sub` keeps the tests debug-buildable without
/// changing which value they test.
#[cfg(test)]
pub(crate) fn prefix_shape(chars: usize) -> (usize, u8) {
    let mut last_byte_idx = (5 * chars / 8).wrapping_sub(1);
    let n_bits = (5 * chars) % 8;
    let last_byte_mask = (((1u32 << n_bits) - 1) << (8 - n_bits)) as u8;
    if last_byte_mask > 0 {
        last_byte_idx += 1;
    }
    (last_byte_idx, last_byte_mask)
}

/// The prefix bytes the host builds for the key `key`: the key's first
/// `5 * chars` bits as the address sees them, with every remaining bit cleared.
///
/// The address is `base32(pubkey || checksum || 0x03)` in RFC 4648 order, which
/// reads the byte string most-significant-bit first, so bit `b` of that stream
/// is bit `7 - (b % 8)` of key byte `b / 8`. Clearing the rest reproduces
/// `base32::decode(s + "aa")` exactly, because the "aa" padding is two zero
/// 5-bit groups. Cross-checked against the real `base32` crate for every length
/// 1..=12 and 2000 random prefixes each; see `TOPBITS.md`.
#[cfg(test)]
pub(crate) fn prefix_bytes_of(key: &[u8; 32], chars: usize) -> [u8; 16] {
    let mut out = [0u8; 16];
    for b in 0..(5 * chars) {
        let bit = (key[b >> 3] >> (7 - (b & 7))) & 1;
        out[b >> 3] |= (bit as u8) << (7 - (b & 7));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustacuda_core::DevicePointer;
    #[allow(unused_imports)]
    use std::vec::Vec;

    /// A deterministic xorshift, so a failure is replayable.
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
        fn byte(&mut self) -> u8 {
            (self.next() >> 24) as u8
        }
    }

    /// A `BytePrefix` over a **host** byte array, built the way `gpu.rs` builds
    /// the device one: same fields, same `prefix_word_masks`, pointers into host
    /// memory (`DevicePointer` is a raw pointer, so the host tests can aim it
    /// anywhere). `out` and `count` are never dereferenced by either matching
    /// method.
    fn host_prefix(bytes: &[u8], last_byte_idx: usize, last_byte_mask: u8) -> BytePrefix {
        let (full_mask, full_val, top_mask, top_val) =
            prefix_word_masks(bytes, last_byte_idx, last_byte_mask);
        BytePrefix {
            byte_prefix: unsafe { DevicePointer::wrap(bytes.as_ptr() as *mut u8) },
            byte_prefix_len: bytes.len(),
            last_byte_idx,
            last_byte_mask,
            out: unsafe { DevicePointer::wrap(core::ptr::null_mut()) },
            count: unsafe { DevicePointer::wrap(core::ptr::null_mut()) },
            full_mask,
            full_val,
            top_mask,
            top_val,
        }
    }

    /// The 32-byte candidate whose low 8 bytes the fast path's `word` stands
    /// for. The remaining 24 bytes are irrelevant to every prefix of at most 12
    /// characters, and are filled with noise so that a `matches` implementation
    /// reaching past byte 7 would be caught.
    fn data_from_word(word: u64, tail: &[u8; 24]) -> [u8; 32] {
        let mut data = [0u8; 32];
        for (i, b) in data[..8].iter_mut().enumerate() {
            *b = (word >> (8 * i)) as u8;
        }
        data[8..].copy_from_slice(tail);
        data
    }

    /// The claim the whole optimisation rests on: for every prefix shape, the
    /// 64-bit compare decides exactly what the byte-wise compare decides.
    ///
    /// Four families of input, because they fail differently:
    ///
    /// * a prefix whose last byte has **no** bits outside the mask, which is what
    ///   every real base32 prefix looks like (the bits past the prefix are
    ///   `base32::decode`'s "aa" padding, i.e. zero), tested at the word that
    ///   the prefix describes. Both paths must say **yes**: this is the case
    ///   that catches an inverted comparison, an off-by-one in the top-bit
    ///   extraction, and a mask applied to the wrong side of the comparison.
    /// * every single-bit flip of that word. A bit inside the mask union must
    ///   turn both verdicts to **no**; a bit outside it must leave both at yes.
    ///   This is the case that catches a mask one bit short or one bit long, and
    ///   a shift that drops or invents a bit.
    /// * the same prefix with **bits set outside the mask**, which `matches`
    ///   rejects outright (it compares the masked data byte against the whole
    ///   prefix byte) and `matches_word` has to reject as well. An
    ///   implementation that masked the right-hand side too would agree on every
    ///   real prefix and disagree here.
    /// * a word with no relationship to the data, where both must say no.
    #[test]
    fn matches_word_is_equivalent_to_matches() {
        const MASKS: [u8; 11] = [
            0x00, 0x01, 0x02, 0x40, 0x80, 0xF8, 0xFE, 0xF0, 0xFF, 0x55, 0xAA,
        ];
        let mut rng = Rng(0x0BAD_C0DE_1234_5678);
        for last_byte_idx in 0..=7usize {
            for &last_byte_mask in MASKS.iter() {
                for case in 0..3 {
                    let mut bytes = [0u8; 16];
                    for b in bytes.iter_mut() {
                        *b = rng.byte();
                    }
                    // `case == 1` leaves bits set outside the mask in the last
                    // byte; the other two are shapes a real prefix can have.
                    if case != 1 {
                        bytes[last_byte_idx] &= last_byte_mask;
                    }
                    let bp = host_prefix(&bytes, last_byte_idx, last_byte_mask);
                    let mut tail = [0u8; 24];
                    for b in tail.iter_mut() {
                        *b = rng.byte();
                    }

                    // The word the prefix's own bytes describe.
                    let mut word = 0u64;
                    for i in 0..8 {
                        word |= (bytes[i] as u64) << (8 * i);
                    }
                    let data = data_from_word(word, &tail);
                    // `matches` compares the masked data byte against the whole
                    // prefix byte, and the data *is* the prefix byte here, so it
                    // accepts exactly when the prefix byte has no bits outside
                    // the mask. `case == 1` keeps such bits, and does so for
                    // every mask narrower than the whole byte.
                    let expect = bytes[last_byte_idx] & !last_byte_mask == 0;
                    assert_eq!(
                        bp.matches(&data),
                        expect,
                        "j {} m {:#04x} case {}: the reference verdict moved",
                        last_byte_idx,
                        last_byte_mask,
                        case
                    );
                    assert_eq!(
                        bp.matches_word(word),
                        expect,
                        "j {} m {:#04x} case {}: the fast path verdict differs",
                        last_byte_idx,
                        last_byte_mask,
                        case
                    );

                    // Every single-bit flip.
                    for bit in 0..64usize {
                        let flipped = word ^ (1u64 << bit);
                        assert_eq!(
                            bp.matches_word(flipped),
                            bp.matches(&data_from_word(flipped, &tail)),
                            "j {} m {:#04x} case {}: the two paths disagree on bit {}",
                            last_byte_idx,
                            last_byte_mask,
                            case,
                            bit
                        );
                    }

                    // A word with no relationship to the data.
                    let other = rng.next();
                    assert_eq!(
                        bp.matches_word(other),
                        bp.matches(&data_from_word(other, &tail)),
                        "j {} m {:#04x} case {}: the two paths disagree on an unrelated word",
                        last_byte_idx,
                        last_byte_mask,
                        case
                    );
                }
            }
        }
    }

    /// The mask union must be *exactly* the set of bits `matches` constrains:
    /// the `8 * last_byte_idx` low bits, plus the `last_byte_mask` bits of the
    /// byte at `last_byte_idx`. One bit too few and the search returns keys that
    /// do not match; one bit too many and it never matches anything. Both are
    /// silent, and both are what an off-by-one in the host's mask arithmetic
    /// looks like.
    #[test]
    fn word_masks_cover_exactly_the_compared_bits() {
        for last_byte_idx in 0..=7usize {
            for &last_byte_mask in [0x00u8, 0x01, 0x80, 0xF8, 0xFE, 0xF0, 0xFF].iter() {
                let mut bytes = [0u8; 16];
                for (i, b) in bytes.iter_mut().enumerate() {
                    *b = 0xA5u8 ^ (i as u8).wrapping_mul(37);
                }
                let bp = host_prefix(&bytes, last_byte_idx, last_byte_mask);
                let k = 8 * last_byte_idx;
                let mut want = 0u64;
                for bit in 0..k {
                    want |= 1u64 << bit;
                }
                // Mask bit `p` is bit `p` of the prefix byte, which is bit
                // `k + p` of the word.
                for p in 0..8 {
                    if last_byte_mask & (1u8 << p) != 0 {
                        want |= 1u64 << (k + p);
                    }
                }
                assert_eq!(
                    bp.full_mask | bp.top_mask,
                    want,
                    "j {} m {:#04x}: the mask union covers the wrong bits",
                    last_byte_idx,
                    last_byte_mask
                );
                // The two halves must not overlap: an overlap would mean the
                // partial byte is being counted twice and a value bit is lost.
                assert_eq!(
                    bp.full_mask & bp.top_mask,
                    0,
                    "j {} m {:#04x}: the two masks overlap",
                    last_byte_idx,
                    last_byte_mask
                );
                // And the whole-byte half carries no value bits above its mask.
                // The partial-byte half deliberately *does* (see
                // `prefix_word_masks`): `matches` compares the masked data byte
                // against the whole prefix byte, so an out-of-mask bit there has
                // to be able to reject.
                assert_eq!(bp.full_val & !bp.full_mask, 0, "j {}: stray value bits", last_byte_idx);
            }
        }
    }

    /// A prefix wider than 64 observable bits has no `u64` to be packed into, so
    /// the host must be told rather than silently truncating. Thirteen base32
    /// characters is the first length that does it: `last_byte_idx` 8 means byte
    /// 8, and bit 64 is the end of the word.
    #[test]
    #[should_panic(expected = "more than 8 bytes")]
    fn prefix_wider_than_the_word_is_rejected() {
        let bytes = [0u8; 16];
        prefix_word_masks(&bytes, 8, 0xF0);
    }

    /// The host fills this struct by value and uploads it, so a field added on one
    /// side only, or a padding hole, would leave the kernel comparing against
    /// masks the host never wrote. `#[repr(C)]` plus these offsets is the
    /// contract, and the four masks are adjacent at the tail so the kernel can
    /// load them with two 64-bit loads.
    #[test]
    fn byte_prefix_layout_is_flat() {
        assert_eq!(core::mem::size_of::<BytePrefix>(), 80);
        assert_eq!(core::mem::align_of::<BytePrefix>(), 8);
        let bp = BytePrefix {
            byte_prefix: unsafe { DevicePointer::wrap(core::ptr::null_mut()) },
            byte_prefix_len: 0,
            last_byte_idx: 0,
            last_byte_mask: 0,
            out: unsafe { DevicePointer::wrap(core::ptr::null_mut()) },
            count: unsafe { DevicePointer::wrap(core::ptr::null_mut()) },
            full_mask: 0,
            full_val: 0,
            top_mask: 0,
            top_val: 0,
        };
        let base = &bp as *const BytePrefix as usize;
        let off = |p: *const u8| p as usize - base;
        assert_eq!(off(&bp.byte_prefix as *const _ as *const u8), 0);
        assert_eq!(off(&bp.byte_prefix_len as *const _ as *const u8), 8);
        assert_eq!(off(&bp.last_byte_idx as *const _ as *const u8), 16);
        assert_eq!(off(&bp.last_byte_mask as *const _ as *const u8), 24);
        assert_eq!(off(&bp.out as *const _ as *const u8), 32);
        assert_eq!(off(&bp.count as *const _ as *const u8), 40);
        assert_eq!(off(&bp.full_mask as *const _ as *const u8), 48);
        assert_eq!(off(&bp.full_val as *const _ as *const u8), 56);
        assert_eq!(off(&bp.top_mask as *const _ as *const u8), 64);
        assert_eq!(off(&bp.top_val as *const _ as *const u8), 72);
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
