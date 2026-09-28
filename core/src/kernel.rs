//! Batched kernel: one SHA-512 and one scalar multiplication per thread,
//! amortised over a whole batch of candidate keys.
//!
//! # What the naive kernel did
//!
//! Per candidate key:
//!
//! ```text
//! seed -> SHA-512 -> [h]B -> compress
//! ```
//!
//! The scalar multiplication dominates. `ed25519-compact`'s
//! `ge_scalarmult_base` is a plain double-and-add over all 256 bits, so it
//! performs **512 point additions per key** (~4096 field multiplies).
//!
//! # What this kernel does
//!
//! Each thread owns an independent *batch* of `CHUNK * 8` candidate keys
//! derived from one seed, and pays the fixed costs once:
//!
//! 1. **One** SHA-512 of the seed, giving a 64-byte expanded key. Its lower
//!    half is the clamped scalar, its upper half is the signing prefix.
//! 2. **One** table-based scalar multiplication, summing precomputed
//!    `[2^i]B` values. A random 253-bit scalar has ~126 set bits, so this is
//!    ~126 additions and, crucially, **no doublings** -- the naive loop
//!    rebuilds that same ladder for every key.
//! 3. Then a walk: repeatedly add the fixed point `8B`, which is one point
//!    addition per additional candidate.
//!
//! Step 3 is the same trick mkp224o uses at `worker_batch.inc.h:52-64`. It is
//! legal for Tor because `hs_ed25519_secret_key` stores the *expanded* key:
//! the pubkey is derived from the scalar, and advancing a clamped scalar by a
//! small amount preserves the ed25519 clamping invariants
//! (`sk[0] & 248 == sk[0]`, `(sk[31] & 63) | 64 == sk[31]`), so every step of
//! the walk yields a valid key.
//!
//! 4. Compression needs `Z^-1`, and an inversion costs ~254 squarings -- more
//!    than the point arithmetic it serves. Candidates are therefore checked in
//!    chunks of `CHUNK`, batch-inverting the chunk with Montgomery's trick so
//!    one inversion serves `CHUNK` points.
//!
//! Each chunk is walked **once**. The batch inversion needs every `Z` before it
//! can serve any of them, which used to mean walking the chunk twice -- once to
//! collect the `Z`s and again to compress the points with the inverses that came
//! back -- for two point additions per candidate where one does. Only the `y`
//! coordinate is needed (see `BatchState::check`), so the walk keeps `Y` next to
//! `Z` and the second pass becomes a multiply and a compare. See `render`.
//!
//! Net effect per candidate: one point addition, one field multiply and
//! `1/CHUNK` of an inversion, versus 512 point additions plus a full inversion.

// Imports used by `BatchState` and the batch maths, which compile for both
// the device and the host (the latter so the unit tests can exercise them).
use core::sync::atomic::{AtomicU32, Ordering};

use super::field::{fe_to_bytes_low8, Fe};
use super::group::{ge_madd, scalarmult_base_comb, BaseTables, Ge, Niels};
use super::sha512::sha512;
use super::{KernelParams, SLOT_BYTES, SLOTS};

// Used only by the `render` entry point, which is device-only.
#[cfg(target_arch = "nvptx64")]
use super::field::{fe_mul, FE_ONE};
#[cfg(target_arch = "nvptx64")]
use super::group::{batch_invert_into, COMB_BITS, COMB_LEN, COMB_W};
#[cfg(target_arch = "nvptx64")]
use super::{BATCH, BLOCK, CHUNK};

/// A match found during a batch.
#[derive(Copy, Clone)]
pub struct Hit {
    /// The 64-byte expanded secret key: clamped scalar || signing prefix.
    pub expanded: [u8; 64],
    /// The compressed public key.
    pub pubkey: [u8; 32],
}

/// Per-thread working state, held in registers across a batch.
///
/// Note there is no local match buffer. There used to be `hits: [Hit; 8]`,
/// which cost 8 * 96 bytes = 192 registers per thread and pushed the kernel to
/// 248 registers -- the hardware ceiling -- leaving room for only 8 warps per
/// SM, about 17% of the 48 an SM can hold. Matches now go straight into the
/// shared capture ring, so the buffer has nothing left to do.
pub struct BatchState {
    /// The walking point: candidate `i` is `cur = [h + 8i]B`.
    cur: Ge,
    /// The fixed step, `8B`, in Niels form.
    ///
    /// This used to be computed per thread by `make_eight` (three doublings of
    /// the base point, 24 field multiplies every thread every launch) and kept
    /// as a full `Ge`. Both are gone: the host uploads `8B` in the same blob as
    /// the comb table, and the walk now costs 7M per step instead of 9M
    /// because `ge_madd` folds `2d*T` into the addend.
    eight: Niels,
    /// Expanded key for the current scalar.
    expanded: [u8; 64],
    /// Candidate counter.
    index: usize,
}

impl BatchState {
    pub const fn new() -> Self {
        BatchState {
            cur: Ge {
                x: [0; 5],
                y: [0; 5],
                z: [0; 5],
                t: [0; 5],
            },
            eight: Niels {
                yplusx: [0; 5],
                yminusx: [0; 5],
                xy2d: [0; 5],
            },
            expanded: [0; 64],
            index: 0,
        }
    }

    /// Pay the fixed costs once: SHA-512, clamping, one scalar mult.
    ///
    /// `base` is the host-built comb table plus `8B`, shared by every thread.
    pub fn begin(&mut self, seed: &[u8; 32], base: &BaseTables) {
        self.expanded = sha512(seed);
        // Clamp the lower half in place; the upper half is the nonce prefix
        // and is left alone.
        self.expanded[0] &= 248;
        self.expanded[31] &= 63;
        self.expanded[31] |= 64;

        let mut scalar = [0u8; 32];
        scalar.copy_from_slice(&self.expanded[..32]);
        self.cur = scalarmult_base_comb(&scalar, &base.comb);
        self.eight = base.eight;
        self.index = 0;
    }

    /// Advance to the next candidate: one point addition, 8 scalar steps.
    #[inline(always)]
    pub fn advance(&mut self) {
        self.cur = ge_madd(&self.cur, &self.eight);
        self.index += 8;
        // The scalar is advanced in lockstep so the expanded key we hand back
        // matches the point we just computed.
        add_clamped_scalar(&mut self.expanded[..32], 8);
    }

    /// Publish a match straight into the prefix's capture ring.
    ///
    /// `y` is the candidate's affine `y` coordinate, `Y/Z` -- **not** a
    /// compressed public key. A Tor onion address is
    /// `base32(pubkey || checksum || version)`, so a prefix is decided by the
    /// low bits of the encoding, which are the low bits of `y`; the sign of `x`
    /// is bit 255, twenty-three bytes above anything a prefix can reach. So
    /// `X/Z` is never computed, and neither is the other 24 bytes of the
    /// encoding. That is what lets the kernel walk each chunk once (see
    /// `render`): `Y` is still in hand when the batch inverse lands, and
    /// `fe_to_bytes_low8` is a reduction and a shift rather than a 32-byte
    /// packing loop.
    ///
    /// The slot index is claimed with a device-scope atomic, so threads that
    /// match the same prefix in the same launch each take a distinct slot
    /// instead of overwriting one another. Once the ring is full the excess is
    /// dropped rather than wrapped, so a captured key is never overwritten by a
    /// later one; the host reads `count` to see the true total and can tell
    /// that it captured fewer.
    ///
    /// Publishing here rather than buffering and flushing later is what keeps
    /// `BatchState` small: the old `[Hit; 8]` buffer was 192 registers a thread.
    pub fn check(&mut self, params: &mut KernelParams, y: &Fe) {
        if params.byte_prefixes_len == 0 {
            return;
        }
        let low = fe_to_bytes_low8(y);
        for i in 0..params.byte_prefixes_len {
            // Mutable: the ring and its cursor are written through it. The
            // device crate offers no aliasing guarantees, hence the raw pointer.
            let bp = unsafe { &mut *params.byte_prefixes.as_raw_mut().add(i) };
            if bp.matches_low8(low) {
                unsafe {
                    // Claim a slot. `count` only ever increases within a
                    // launch, so every claim gets a distinct index.
                    let idx = atomic_add(bp.count.as_raw_mut(), 1) as usize;
                    if idx < SLOTS {
                        let out = bp.out.as_raw_mut().add(idx * SLOT_BYTES);
                        for k in 0..SLOT_BYTES {
                            *out.add(k) = self.expanded[k];
                        }
                    }
                }
                return;
            }
        }
    }
}

/// `atom.relaxed.sys.add.u32`, returning the pre-increment value.
///
/// `Relaxed` is sufficient: the returned index only has to be unique, each
/// thread writes solely to the slot it claimed, and the host reads the ring
/// after the launch completes. LLVM lowers this to a real PTX atomic (verified:
/// `atom.relaxed.sys.add.u32`), so no inline asm is needed.
#[inline(always)]
fn atomic_add(ptr: *mut u32, val: u32) -> u32 {
    unsafe { (*(ptr as *const AtomicU32)).fetch_add(val, Ordering::Relaxed) }
}

/// Add a small delta to a clamped 32-byte scalar, preserving the clamping
/// invariants that make it a valid ed25519 secret scalar.
///
/// Mirrors mkp224o's `addsztoscalar32`; it walks forward with an explicit
/// carry and re-clamps, which is a no-op for the small deltas used here.
pub fn add_clamped_scalar(scalar: &mut [u8], delta: u64) {
    let mut carry = delta;
    for byte in scalar.iter_mut() {
        if carry == 0 {
            break;
        }
        let sum = *byte as u64 + (carry & 0xFF);
        *byte = sum as u8;
        carry = (carry >> 8) + (sum >> 8);
    }
    scalar[0] &= 248;
    scalar[31] &= 63;
    scalar[31] |= 64;
}

/// The `render` entry point the host launches via `KERNEL_PTX_PATH`.
///
/// Args: `params` and the host-built comb table (plus `8B`).
#[cfg(target_arch = "nvptx64")]
#[no_mangle]
pub extern "ptx-kernel" fn render(params_ptr: *mut KernelParams, base_ptr: *const BaseTables) {
    use core::arch::nvptx::*;

    let params = unsafe { &mut *params_ptr };
    let base = unsafe { &*base_ptr };
    // Guard the compile-time tunables. A zero or misordered value would make the
    // chunk buffers zero-length and the walk silently do nothing, which looks
    // like a mysteriously slow kernel rather than a bad setting.
    assert!(CHUNK > 0, "T3V_CHUNK must be greater than zero");
    assert!(BATCH > 0, "T3V_BATCH must be greater than zero");
    assert!(
        BATCH % CHUNK == 0,
        "T3V_BATCH ({}) must be a multiple of T3V_CHUNK ({})",
        BATCH,
        CHUNK
    );
    assert!(COMB_W >= 2 && COMB_W <= 4, "T3V_COMB_W must be in 2..=4");
    assert!(COMB_BITS % COMB_W == 0, "T3V_COMB_W must divide 252");
    assert!(COMB_LEN > 0, "the comb table must not be empty");

    let tid = unsafe { _block_dim_x() } * unsafe { _block_idx_x() } + unsafe { _thread_idx_x() };
    let tid = tid as u64;

    // Each thread derives its own seed from the block's base seed, so the
    // batch covers BATCH * BLOCK distinct candidates.
    let params_seed = unsafe { core::slice::from_raw_parts(params.seed.as_raw(), 32) };
    let mut seed = [0u8; 32];
    seed.copy_from_slice(params_seed);
    add_u256(&mut seed, tid);

    let mut state = BatchState::new();
    state.begin(&seed, base);

    // With no prefixes there is nothing to match, so skip matching entirely:
    // no compression, no batch inversion, no second pass. This is what
    // `--bench` uses to measure pure key generation, and it is why that number
    // is an upper bound on search throughput rather than the search rate.
    if params.byte_prefixes_len == 0 {
        for _ in 0..(BATCH / CHUNK) {
            for _ in 0..CHUNK {
                state.advance();
            }
        }
        return;
    }

    // Walk in chunks, batch-inverting each chunk's Z coordinates so one
    // inversion serves CHUNK points.
    //
    // Each chunk is walked **once**. The batch inversion needs every Z before
    // it can serve any of them, so this used to walk the chunk twice: once to
    // collect the Z coordinates, and once more to compress and test the points
    // with the inverses that came back. That is two point additions per
    // candidate where one does, i.e. 14 field multiplies instead of 7.
    //
    // The re-walk is only needed because compression divides by Z, and the
    // dividend `Y` has to be in hand at the same time as the divisor. So keep
    // it: `ys` holds each candidate's Y while `zs` holds its Z, and after the
    // inversion `y_i = Y_i * Z_i^-1` is one multiply away. The cost is
    // CHUNK * 5 u64 of local memory (10,240 B/thread at CHUNK=256) against one
    // point addition -- 7 multiplies and 5 add/sub chains -- per candidate.
    //
    // Buffering the whole point instead would cost CHUNK * 20 u64 and destroy
    // occupancy, which is why the re-walk existed in the first place; keeping
    // only the two coordinates compression needs is 4x less than that.
    let mut ys = [FE_ONE; CHUNK];
    let mut zs = [FE_ONE; CHUNK];
    let mut scratch = [FE_ONE; CHUNK];

    for _ in 0..(BATCH / CHUNK) {
        // Pass 1: walk the chunk, keeping each candidate's Y and Z.
        //
        // Storing *before* stepping is what leaves `cur` on the first candidate
        // of the next chunk: the last iteration stores this chunk's last
        // candidate and then takes the step past it. Storing after stepping
        // would leave `cur` on the last candidate of this chunk, and the next
        // chunk would test it again and drop the batch's final candidate.
        //
        // The scalar is deliberately *not* advanced here. It advances once per
        // candidate in pass 2, so when the inversion lands it is still the
        // chunk's first candidate's.
        for c in 0..CHUNK {
            ys[c] = state.cur.y;
            zs[c] = state.cur.z;
            state.cur = ge_madd(&state.cur, &state.eight);
        }

        // One inversion serves the whole chunk: `zs[c]` becomes `Z^-1`.
        batch_invert_into(&mut zs, &mut scratch);

        // Pass 2: no point arithmetic at all. Every Y is still in hand, so each
        // candidate is compressed from the batch inverse and tested.
        for c in 0..CHUNK {
            let y = fe_mul(&ys[c], &zs[c]);
            state.check(&mut *params, &y);
            // Advance the scalar in lockstep with the walk above, so the key
            // `check` hands back is the one just tested and the next chunk
            // starts from the right candidate. Advancing after each test leaves
            // the key on the first candidate of the next chunk.
            add_clamped_scalar(&mut state.expanded[..32], 8);
        }
        state.index += 8 * CHUNK;
    }
}

/// Add a 64-bit delta to a 32-byte little-endian integer, without clamping.
///
/// Gives each thread in the block a distinct starting seed, so the launch
/// covers `BATCH * BLOCK` distinct candidate keys. Only the device entry point
/// needs it; the host build keeps it alive for the tests below.
#[cfg_attr(not(test), allow(dead_code))]
fn add_u256(v: &mut [u8; 32], delta: u64) {
    let mut carry = delta;
    for byte in v.iter_mut() {
        if carry == 0 {
            break;
        }
        let sum = *byte as u64 + (carry & 0xFF);
        *byte = sum as u8;
        carry = (carry >> 8) + (sum >> 8);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[allow(dead_code)]
    use super::add_u256;
    #[allow(unused_imports)]
    use std::vec::Vec;
    #[allow(unused_imports)]
    use std::string::String;
    // `render` is device-only, so its imports are cfg'd to the device build.
    // The tests below exercise the same maths, so pull in the ones they need.
    use crate::field::{fe_invert, fe_mul, fe_to_bytes_low8, FE_ONE};
    use crate::group::{
        batch_invert_into, build_power_table, ge_to_bytes_with_zi, scalarmult_base_table,
        TABLE_LEN,
    };
    use crate::{BATCH, CHUNK};

    /// The host-built blob the kernel actually receives, plus the power table
    /// that `scalarmult_base_table` needs as the reference implementation.
    struct Tables {
        base: BaseTables,
        power: [Ge; TABLE_LEN],
    }

    fn tables() -> Tables {
        let zero = Ge {
            x: [0; 5],
            y: [0; 5],
            z: [0; 5],
            t: [0; 5],
        };
        let mut power = [zero; TABLE_LEN];
        build_power_table(&mut power);

        let mut base: BaseTables = unsafe { core::mem::zeroed() };
        crate::group::build_comb_table(&mut base.comb, &mut base.eight);
        Tables { base, power }
    }

    fn unhex(s: &str) -> [u8; 32] {
        let mut out = [0u8; 32];
        for i in 0..32 {
            out[i] = u8::from_str_radix(&s[i * 2..i * 2 + 2], 16).unwrap();
        }
        out
    }

    fn hex(b: &[u8]) -> String {
        b.iter().map(|x| format!("{:02x}", x)).collect()
    }

    /// The pubkey for a given clamped scalar, computed the slow way.
    fn pubkey_of_scalar(scalar: &[u8; 32], power: &[Ge; TABLE_LEN]) -> [u8; 32] {
        let p = scalarmult_base_table(scalar, power);
        let zi = fe_invert(&p.z);
        ge_to_bytes_with_zi(&p, &zi)
    }

    #[test]
    fn begin_matches_rfc8032() {
        // The first candidate of a batch must be the pubkey of the seed's
        // clamped scalar. This is the exact path the kernel takes.
        let t = tables();
        let seed = unhex("9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60");

        let mut st = BatchState::new();
        st.begin(&seed, &t.base);

        let zi = fe_invert(&st.cur.z);
        let pk = ge_to_bytes_with_zi(&st.cur, &zi);
        assert_eq!(
            hex(&pk),
            "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a"
        );
    }

    #[test]
    fn walk_matches_incremental_scalarmults() {
        // The whole point of the batch: candidate i is [h + 8i]B. Verify that
        // for a full batch, which is exactly what the kernel checks against.
        let t = tables();
        let seed = unhex("4ccd089b28ff96da9db6c346ec114e0f5b8a319f35aba624da8cf6ed4fb8a6fb");

        let mut st = BatchState::new();
        st.begin(&seed, &t.base);

        // Candidate 0 is the seed's own scalar.
        let mut scalar = [0u8; 32];
        scalar.copy_from_slice(&st.expanded[..32]);
        let want = pubkey_of_scalar(&scalar, &t.power);
        let zi = fe_invert(&st.cur.z);
        assert_eq!(hex(&ge_to_bytes_with_zi(&st.cur, &zi)), hex(&want));

        // Every later candidate must equal a fresh scalarmult of scalar + 8i.
        for i in 1..BATCH {
            st.advance();
            assert_eq!(
                st.expanded[..32],
                {
                    let mut s = [0u8; 32];
                    s.copy_from_slice(&st.expanded[..32]);
                    s
                }[..],
                "expanded key is the scalar"
            );
            let zi = fe_invert(&st.cur.z);
            let got = ge_to_bytes_with_zi(&st.cur, &zi);
            let want = pubkey_of_scalar(&{
                let mut s = [0u8; 32];
                s.copy_from_slice(&st.expanded[..32]);
                s
            }, &t.power);
            assert_eq!(
                hex(&got),
                hex(&want),
                "candidate {} must be [h+{}]B",
                i,
                i * 8
            );
        }
    }

    /// A `BatchState` plus the two per-candidate outputs the kernel's chunk
    /// loop produces, so the loop can be written once and driven by the tests.
    ///
    /// The kernel itself calls `BatchState::check`, which needs device pointers
    /// into a prefix list. The tests cannot provide those, so they capture what
    /// `check` is given instead: the low word it compares, and the expanded key
    /// it would have published.
    struct Walked {
        st: BatchState,
        y_low: Vec<u64>,
        expanded: Vec<[u8; 64]>,
    }

    impl Walked {
        fn new(seed: &[u8; 32], t: &Tables) -> Self {
            let mut st = BatchState::new();
            st.begin(seed, &t.base);
            Walked {
                st,
                y_low: Vec::new(),
                expanded: Vec::new(),
            }
        }
    }

    /// The kernel's chunk loop, verbatim.
    ///
    /// Written out here rather than factored into a shared function so this
    /// test exercises the same statements the device runs, and so it cannot
    /// drift from `render` in the part that matters: which scalar goes with
    /// which point, and which candidates are tested at all.
    fn chunk_walk(w: &mut Walked, ys: &mut [Fe], zs: &mut [Fe], scratch: &mut [Fe]) {
        for _ in 0..(BATCH / CHUNK) {
            for c in 0..CHUNK {
                ys[c] = w.st.cur.y;
                zs[c] = w.st.cur.z;
                w.st.cur = ge_madd(&w.st.cur, &w.st.eight);
            }

            batch_invert_into(zs, scratch);

            for c in 0..CHUNK {
                let y = fe_mul(&ys[c], &zs[c]);
                // Exactly what `check` computes, and the key it would publish.
                w.y_low.push(fe_to_bytes_low8(&y));
                w.expanded.push(w.st.expanded);
                add_clamped_scalar(&mut w.st.expanded[..32], 8);
            }
            w.st.index += 8 * CHUNK;
        }
    }

    #[test]
    fn chunked_walk_matches_sequential_walk() {
        // Each chunk is walked once now, keeping Y and Z. The y the second pass
        // derives from them must be the same affine y the sequential reference
        // compresses, and the expanded key must be in lockstep with it -- the
        // key handed back on a match has to be the one that produced the point.
        let t = tables();
        let seed = unhex("c5aa8df43f9f837bedb7442f31dcb7b166d38535076f094b85ce3a2e0b4458f7");

        let mut got = Walked::new(&seed, &t);
        let mut ys = [FE_ONE; CHUNK];
        let mut zs = [FE_ONE; CHUNK];
        let mut scratch = [FE_ONE; CHUNK];
        chunk_walk(&mut got, &mut ys, &mut zs, &mut scratch);

        // Sequential reference: pubkey of every candidate, one inversion each.
        let mut reference = BatchState::new();
        reference.begin(&seed, &t.base);
        assert_eq!(got.y_low.len(), BATCH);
        for i in 0..BATCH {
            let zi = fe_invert(&reference.cur.z);
            let y = fe_mul(&reference.cur.y, &zi);
            assert_eq!(
                got.y_low[i],
                fe_to_bytes_low8(&y),
                "candidate {}: the batch y must be the sequential y",
                i
            );
            // The whole pubkey agrees, which pins the point as well as the word.
            let pk = ge_to_bytes_with_zi(&reference.cur, &zi);
            assert_eq!(
                got.y_low[i] as u32,
                u32::from_le_bytes([pk[0], pk[1], pk[2], pk[3]]),
                "candidate {}: low four bytes of the encoding",
                i
            );
            // And the expanded key is the one belonging to this candidate.
            assert_eq!(
                &got.expanded[i][..],
                &reference.expanded[..],
                "candidate {}: expanded key must be in lockstep",
                i
            );
            reference.advance();
        }

        // After the batch, the state is on the same candidate either way.
        assert_eq!(reference.cur.y, got.st.cur.y);
        assert_eq!(reference.cur.z, got.st.cur.z);
        assert_eq!(reference.expanded, got.st.expanded);
        assert_eq!(reference.index, got.st.index);
    }

    #[test]
    fn y_word_is_enough_to_decide_every_prefix() {
        // The load-bearing claim: the low eight bytes of y decide a prefix
        // exactly, for every prefix length the host can build, with no
        // dependence on anything else in the encoding. Checked over real
        // pubkeys from a real walk, and against `BytePrefix::matches` on the
        // full 32-byte key as the reference.
        let t = tables();
        let seed = unhex("4ccd089b28ff96da9db6c346ec114e0f5b8a319f35aba624da8cf6ed4fb8a6fb");
        let mut st = BatchState::new();
        st.begin(&seed, &t.base);

        let mut keys: Vec<([u8; 32], u64)> = Vec::new();
        for _ in 0..64 {
            let zi = fe_invert(&st.cur.z);
            let y = fe_mul(&st.cur.y, &zi);
            keys.push((ge_to_bytes_with_zi(&st.cur, &zi), fe_to_bytes_low8(&y)));
            st.advance();
        }

        for l in 1..=crate::BytePrefix::MAX_PREFIX_CHARS {
            // Build a prefix from one key, the way the host does: the key's
            // first bytes with the bits the partial mask clears zeroed.
            let mut last_byte_idx = (5 * l / 8) - 1;
            let n_bits = (5 * l) % 8;
            let last_byte_mask = ((1u8 << n_bits) - 1) << (8 - n_bits);
            if last_byte_mask > 0 {
                last_byte_idx += 1;
            }
            let mut slice = keys[0].0[..last_byte_idx + 1].to_vec();
            slice[last_byte_idx] &= last_byte_mask;
            assert!(last_byte_idx < 8, "prefix of {} must fit the word", l);

            for (pk, low) in keys.iter() {
                let want = pk[..last_byte_idx] == slice[..last_byte_idx]
                    && pk[last_byte_idx] & last_byte_mask == slice[last_byte_idx];
                let mut bp = crate::BytePrefix {
                    byte_prefix: unsafe {
                        rustacuda_core::DevicePointer::wrap(slice.as_ptr() as *mut u8)
                    },
                    byte_prefix_len: slice.len(),
                    last_byte_idx,
                    last_byte_mask,
                    prefix_word: crate::BytePrefix::pack_prefix_word(&slice),
                    out: rustacuda_core::DevicePointer::null(),
                    count: rustacuda_core::DevicePointer::null(),
                };
                assert_eq!(bp.matches(pk), want, "matches(), l={}", l);
                assert_eq!(bp.matches_low8(*low), want, "matches_low8(), l={}", l);
                // The sign bit must not matter -- that is what licenses the
                // kernel to skip X/Z entirely.
                let mut flipped = *pk;
                flipped[31] ^= 0x80;
                assert_eq!(bp.matches_low8(*low), want, "sign bit, l={}", l);
                bp.last_byte_mask = last_byte_mask;
                assert_eq!(bp.matches(&flipped), want, "flipped key, l={}", l);
            }
        }
    }

    #[test]
    fn expanded_key_is_independent_of_the_pubkey() {
        // Claim 3: the key reported on a match is the scalar plus the nonce
        // prefix, with no dependence on the compression path. So the expanded
        // keys the walk produces must be exactly the clamped scalars, and must
        // be the same whether they are read during the chunked walk (where only
        // Y and Z were kept) or from a plain sequential walk.
        let t = tables();
        let seed = unhex("9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60");

        let mut st = BatchState::new();
        st.begin(&seed, &t.base);
        let mut want = Vec::new();
        for _ in 0..BATCH {
            want.push(st.expanded);
            st.advance();
        }

        let mut got = Walked::new(&seed, &t);
        let mut ys = [FE_ONE; CHUNK];
        let mut zs = [FE_ONE; CHUNK];
        let mut scratch = [FE_ONE; CHUNK];
        chunk_walk(&mut got, &mut ys, &mut zs, &mut scratch);

        assert_eq!(got.expanded, want);
        // And each one is a valid clamped scalar whose pubkey is the candidate's.
        for (i, expanded) in want.iter().enumerate() {
            assert_eq!(expanded[0] & 7, 0, "candidate {}", i);
            assert_eq!(expanded[31] & 0xC0, 0x40, "candidate {}", i);
        }
    }

    #[test]
    fn clamped_scalar_invariants_hold_throughout() {
        // Every expanded key the kernel could hand back must satisfy the
        // ed25519 clamping invariants, or Tor would reject it.
        let t = tables();
        let seed = unhex("9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60");
        let mut st = BatchState::new();
        st.begin(&seed, &t.base);
        for i in 0..BATCH {
            let s = &st.expanded[..32];
            assert_eq!(s[0] & 7, 0, "candidate {}: low 3 bits must be clear", i);
            assert_eq!(s[31] & 0x80, 0, "candidate {}: bit 255 clear", i);
            assert_eq!(s[31] & 0x40, 0x40, "candidate {}: bit 254 set", i);
            st.advance();
        }
    }

    #[test]
    fn add_u256_wraps_without_clamping() {
        let mut v = [0xFFu8; 32];
        add_u256(&mut v, 1);
        assert_eq!(v, [0u8; 32], "0xFF..FF + 1 wraps to zero");

        let mut v = [0u8; 32];
        v[0] = 0xFF;
        add_u256(&mut v, 1);
        assert_eq!(v[0], 0);
        assert_eq!(v[1], 1);
    }
}
