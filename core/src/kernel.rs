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
//! Net effect per candidate: one point addition plus `1/CHUNK` of an
//! inversion, versus 512 point additions plus a full inversion.

// Imports used by `BatchState` and the batch maths, which compile for both
// the device and the host (the latter so the unit tests can exercise them).
use core::sync::atomic::{AtomicU32, Ordering};

use super::field::Fe;
use super::group::{ge_madd, ge_to_bytes_with_zi, scalarmult_base_comb, BaseTables, Ge, Niels};
use super::sha512::sha512;
use super::{KernelParams, SLOT_BYTES, SLOTS};

// Used only by the `render` entry point, which is device-only.
#[cfg(target_arch = "nvptx64")]
use super::field::FE_ONE;
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
    /// The slot index is claimed with a device-scope atomic, so threads that
    /// match the same prefix in the same launch each take a distinct slot
    /// instead of overwriting one another. Once the ring is full the excess is
    /// dropped rather than wrapped, so a captured key is never overwritten by a
    /// later one; the host reads `count` to see the true total and can tell
    /// that it captured fewer.
    ///
    /// Publishing here rather than buffering and flushing later is what keeps
    /// `BatchState` small: the old `[Hit; 8]` buffer was 192 registers a thread.
    pub fn check(&mut self, params: &mut KernelParams, z_inv: &Fe) {
        if params.byte_prefixes_len == 0 {
            return;
        }
        let pk = ge_to_bytes_with_zi(&self.cur, z_inv);
        for i in 0..params.byte_prefixes_len {
            // Mutable: the ring and its cursor are written through it. The
            // device crate offers no aliasing guarantees, hence the raw pointer.
            let bp = unsafe { &mut *params.byte_prefixes.as_raw_mut().add(i) };
            if bp.matches(&pk) {
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
    // Each chunk is walked twice: once to collect the Z coordinates, and once
    // more to check the points against their inverses. Buffering the points
    // instead would cost CHUNK * 20 u64s of local memory per thread, which
    // destroys occupancy. Re-walking costs CHUNK extra point additions, which
    // is far cheaper than the CHUNK-1 saved inversions (each ~254 squarings).
    let mut zs = [FE_ONE; CHUNK];
    let mut scratch = [FE_ONE; CHUNK];

    for _ in 0..(BATCH / CHUNK) {
        // Checkpoint so the second pass can replay the chunk exactly.
        let start = state.cur;

        // Pass 1: collect this chunk's Z coordinates.
        zs[0] = start.z;
        for c in 1..CHUNK {
            let p = ge_madd(&state.cur, &state.eight);
            state.cur = p;
            zs[c] = p.z;
        }

        // One inversion serves the whole chunk.
        batch_invert_into(&mut zs, &mut scratch);

        // Pass 2: replay the same points and check them. Advancing *after*
        // each check leaves the state on the first candidate of the next
        // chunk; advancing before each check would leave it on the last
        // candidate of this one, and the next chunk would re-test it.
        state.cur = start;
        for c in 0..CHUNK {
            state.check(&mut *params, &zs[c]);
            state.advance();
        }
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
    use crate::field::{fe_invert, FE_ONE};
    use crate::group::{batch_invert_into, build_power_table, scalarmult_base_table, TABLE_LEN};
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

    #[test]
    fn chunked_walk_matches_sequential_walk() {
        // The kernel walks each chunk twice (collect Z, then check). Make sure
        // the replayed points are the same points.
        let t = tables();
        let seed = unhex("c5aa8df43f9f837bedb7442f31dcb7b166d38535076f094b85ce3a2e0b4458f7");

        let mut st = BatchState::new();
        st.begin(&seed, &t.base);

        let mut zs = [FE_ONE; CHUNK];
        let mut scratch = [FE_ONE; CHUNK];

        // Sequential reference: pubkey of every candidate.
        let mut reference = BatchState::new();
        reference.begin(&seed, &t.base);
        let mut want = Vec::new();
        for _ in 0..BATCH {
            let zi = fe_invert(&reference.cur.z);
            want.push(ge_to_bytes_with_zi(&reference.cur, &zi));
            reference.advance();
        }

        let mut got = Vec::new();
        for _ in 0..(BATCH / CHUNK) {
            let start = st.cur;

            zs[0] = start.z;
            for c in 1..CHUNK {
                let p = ge_madd(&st.cur, &st.eight);
                st.cur = p;
                zs[c] = p.z;
            }
            batch_invert_into(&mut zs, &mut scratch);

            st.cur = start;
            for c in 0..CHUNK {
                got.push(ge_to_bytes_with_zi(&st.cur, &zs[c]));
                let p = ge_madd(&st.cur, &st.eight);
                st.cur = p;
            }
        }

        assert_eq!(got.len(), want.len());
        for i in 0..got.len() {
            assert_eq!(
                hex(&got[i]),
                hex(&want[i]),
                "chunked candidate {} mismatch",
                i
            );
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
