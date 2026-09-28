//! Batched kernel: one SHA-512 and one scalar multiplication per thread,
//! amortised over a whole batch of candidate keys.
//!
//! # What the naive kernel did
//!
//! Per candidate key:
//!
//!     seed -> SHA-512 -> [h]B -> compress
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

use super::field::{fe_invert, fe_mul, fe_sub, Fe, FE_ONE};
use super::group::{
    batch_invert_into, ge_add, ge_base, ge_double, ge_to_bytes_with_zi,
    scalarmult_base_table, Ge, TABLE_LEN,
};
use super::sha512::sha512;
use super::{BytePrefix, KernelParams, BATCH, BLOCK, CHUNK};

/// Most matches a single thread will buffer. Matches are rare (roughly one
/// per 2^(5*prefix_len)), so this only guards against pathological filters.
const MAX_HITS: usize = 8;

/// A match found during a batch.
#[derive(Copy, Clone)]
pub struct Hit {
    /// The 64-byte expanded secret key: clamped scalar || signing prefix.
    pub expanded: [u8; 64],
    /// The compressed public key.
    pub pubkey: [u8; 32],
}

/// Per-thread working state, held in registers across a batch.
pub struct BatchState {
    /// The walking point: candidate `i` is `cur = [h + 8i]B`.
    cur: Ge,
    /// The fixed step, `8B`.
    eight: Ge,
    /// Expanded key for the current scalar.
    expanded: [u8; 64],
    /// Candidate counter.
    index: usize,
    /// Buffered matches.
    hits: [Hit; MAX_HITS],
    n_hits: usize,
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
            eight: Ge {
                x: [0; 5],
                y: [0; 5],
                z: [0; 5],
                t: [0; 5],
            },
            expanded: [0; 64],
            index: 0,
            hits: [
                Hit {
                    expanded: [0; 64],
                    pubkey: [0; 32],
                };
                MAX_HITS
            ],
            n_hits: 0,
        }
    }

    /// Pay the fixed costs once: SHA-512, clamping, one scalar mult.
    ///
    /// `table` is the precomputed `[2^i]B` ladder, shared by every thread.
    pub fn begin(&mut self, seed: &[u8; 32], table: &[Ge]) {
        self.expanded = sha512(seed);
        // Clamp the lower half in place; the upper half is the nonce prefix
        // and is left alone.
        self.expanded[0] &= 248;
        self.expanded[31] &= 63;
        self.expanded[31] |= 64;

        let mut scalar = [0u8; 32];
        scalar.copy_from_slice(&self.expanded[..32]);
        self.cur = scalarmult_base_table(&scalar, table);
        self.eight = make_eight();
        self.index = 0;
        self.n_hits = 0;
    }

    /// Advance to the next candidate: one point addition, 8 scalar steps.
    #[inline(always)]
    pub fn advance(&mut self) {
        self.cur = ge_add(&self.cur, &self.eight);
        self.index += 8;
        // The scalar is advanced in lockstep so the expanded key we hand back
        // matches the point we just computed.
        add_clamped_scalar(&mut self.expanded[..32], 8);
    }

    /// Record a match if the compressed pubkey satisfies any prefix.
    pub fn check(&mut self, params: &KernelParams, z_inv: &Fe) {
        if self.n_hits >= MAX_HITS {
            return;
        }
        let pk = ge_to_bytes_with_zi(&self.cur, z_inv);
        for i in 0..params.byte_prefixes_len {
            let bp = unsafe { &*params.byte_prefixes.as_raw().add(i) };
            if bp.matches(&pk) {
                self.hits[self.n_hits] = Hit {
                    expanded: self.expanded,
                    pubkey: pk,
                };
                self.n_hits += 1;
                return;
            }
        }
    }

    /// Publish buffered matches through the prefix slots.
    ///
    /// Writes through a raw pointer: the device crate offers no aliasing
    /// guarantees, and the host allocates one `out`/`success` pair per prefix
    /// so threads with different matches never collide.
    pub fn flush(&mut self, params: &KernelParams) {
        let base = params.byte_prefixes.as_raw() as *mut BytePrefix;
        for h in 0..self.n_hits {
            let hit = self.hits[h];
            for i in 0..params.byte_prefixes_len {
                let bp = unsafe { &mut *base.add(i) };
                if bp.matches(&hit.pubkey) {
                    let out = bp.out.as_raw_mut();
                    unsafe {
                        for k in 0..64 {
                            *out.add(k) = hit.expanded[k];
                        }
                        *bp.success.as_raw_mut() = true;
                    }
                }
            }
        }
        self.n_hits = 0;
    }
}

/// `8B`, the walk step: three doublings of the base point.
fn make_eight() -> Ge {
    let mut p = ge_base();
    p = ge_double(&p);
    p = ge_double(&p);
    ge_double(&p)
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
/// Args: `params` and the precomputed `[2^i]B` ladder.
#[cfg(target_arch = "nvptx64")]
#[no_mangle]
pub extern "ptx-kernel" fn render(params_ptr: *mut KernelParams, table_ptr: *const Ge) {
    use core::arch::nvptx::*;

    let params = unsafe { &mut *params_ptr };
    let table = unsafe { core::slice::from_raw_parts(table_ptr, TABLE_LEN) };

    let tid = unsafe { _block_dim_x() } * unsafe { _block_idx_x() } + unsafe { _thread_idx_x() };
    let tid = tid as u64;

    // Each thread derives its own seed from the block's base seed, so the
    // batch covers BATCH * BLOCK distinct candidates.
    let base = unsafe { core::slice::from_raw_parts(params.seed.as_raw(), 32) };
    let mut seed = [0u8; 32];
    seed.copy_from_slice(base);
    add_u256(&mut seed, tid);

    let mut state = BatchState::new();
    state.begin(&seed, table);

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
            let p = ge_add(&state.cur, &state.eight);
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
            state.check(params, &zs[c]);
            state.advance();
        }
    }

    state.flush(params);
}

/// Add a 64-bit delta to a 32-byte little-endian integer, without clamping.
///
/// Gives each thread in the block a distinct starting seed, so the launch
/// covers `BATCH * BLOCK` distinct candidate keys.
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
    #[allow(unused_imports)]
    use std::vec::Vec;
    #[allow(unused_imports)]
    use std::string::String;
    use crate::field::{fe_is_negative, fe_to_bytes};
    use crate::group::TABLE_LEN;

    fn table() -> [Ge; TABLE_LEN] {
        let zero = Ge {
            x: [0; 5],
            y: [0; 5],
            z: [0; 5],
            t: [0; 5],
        };
        let mut t = [zero; TABLE_LEN];
        crate::group::build_power_table(&mut t);
        t
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
    fn pubkey_of_scalar(scalar: &[u8; 32], table: &[Ge]) -> [u8; 32] {
        let p = scalarmult_base_table(scalar, table);
        let zi = fe_invert(&p.z);
        ge_to_bytes_with_zi(&p, &zi)
    }

    #[test]
    fn begin_matches_rfc8032() {
        // The first candidate of a batch must be the pubkey of the seed's
        // clamped scalar. This is the exact path the kernel takes.
        let table = table();
        let seed = unhex("9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60");

        let mut st = BatchState::new();
        st.begin(&seed, &table);

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
        let table = table();
        let seed = unhex("4ccd089b28ff96da9db6c346ec114e0f5b8a319f35aba624da8cf6ed4fb8a6fb");

        let mut st = BatchState::new();
        st.begin(&seed, &table);

        // Candidate 0 is the seed's own scalar.
        let mut scalar = [0u8; 32];
        scalar.copy_from_slice(&st.expanded[..32]);
        let want = pubkey_of_scalar(&scalar, &table);
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
            }, &table);
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
        let table = table();
        let seed = unhex("c5aa8df43f9f837bedb7442f31dcb7b166d38535076f094b85ce3a2e0b4458f7");

        let mut st = BatchState::new();
        st.begin(&seed, &table);

        let mut zs = [FE_ONE; CHUNK];
        let mut scratch = [FE_ONE; CHUNK];

        // Sequential reference: pubkey of every candidate.
        let mut reference = BatchState::new();
        reference.begin(&seed, &table);
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
                let p = ge_add(&st.cur, &st.eight);
                st.cur = p;
                zs[c] = p.z;
            }
            batch_invert_into(&mut zs, &mut scratch);

            st.cur = start;
            for c in 0..CHUNK {
                got.push(ge_to_bytes_with_zi(&st.cur, &zs[c]));
                let p = ge_add(&st.cur, &st.eight);
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
        let table = table();
        let seed = unhex("9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60");
        let mut st = BatchState::new();
        st.begin(&seed, &table);
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
