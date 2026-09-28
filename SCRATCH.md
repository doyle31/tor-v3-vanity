# `SCRATCH.md` — halving `scratch` in Montgomery batch inversion

**Verdict: the trade is NOT favourable at `CHUNK=256`, and it never becomes
favourable at any `CHUNK`.** The committed change is correct, tested, and
halves `scratch`, but it costs strictly more instructions than it saves, and
the quantity it saves (local-memory *capacity*) is not the binding constraint
at this `CHUNK`. Details and arithmetic below.

The one genuinely surprising result is that `ptxas` drops from 255 to 168
registers when the peel loop is reshaped (§5). That is a real occupancy change,
and — measured, not assumed — it comes from the loop *shape* rather than from
the smaller buffer, so it may be obtainable without paying the extra
multiplies. Worth chasing on its own.

---

## 1. What the change is

`core/src/group.rs`, `batch_invert_stride2_into`: the same Montgomery trick,
but `scratch[j]` holds only the *even-indexed* prefix products.

`batch_invert_into` stores `scratch[i] = P[i-1]` for every `i`, where
`P[i] = z[0]·z[1]·…·z[i]` and `P[-1] = 1`. The backwards peel maintains
`A = 1/P[i]` and does

    out[i] = P[i-1] * A = P[i-1]/P[i] = z[i]^-1
    A     <- A * z[i]                       (which is 1/P[i-1])

so it needs `P[i-1]` at step `i`. Since `P[i-1] = P[i-2] * z[i-1]` and the
peel runs *backwards* — so `z[i-1]` is still the original value when step `i`
runs — keeping only every other prefix costs exactly one `fe_mul` to rebuild
the missing one:

    i even  =>  P[i-1] has odd index, so it *is* scratch[i/2]      (free)
    i odd   =>  P[i-1] = scratch[i/2] * z[i-1]                    (1 mul)

`render` declares `scratch: [Fe; stride2_scratch_len(CHUNK)]`, i.e. 128 slots
instead of 256.

`batch_invert_into` is kept, still tested, and is the oracle the new
implementation is checked against.

### Why not delete `scratch` entirely

`scratch` cannot be removed, only shrunk, and this is a structural fact rather
than an implementation accident. Two arrays is not redundant:

* the chunk's `zs` holds `z[i]`, and pass 2 reads `1/z[i]` back out of it;
* the peel needs `P[i-1]` **and** `z[i]` at every step.

Keeping only the `P` values and re-deriving `z[i]` needs a division (a field
division is an inversion). Keeping only the `z` values and re-deriving `P[i-1]`
needs a fresh product tree — the earlier agent's approach, which costs the
*same* `n/2` extra multiplies as the stride-2 trick while additionally
relying on recursion that LLVM will not inline. So stride-2 is the better of
the two ways to spend the multiplies: same instruction cost, and it keeps the
loop inlinable and branch-light.

---

## 2. `render` SASS, before and after

Built host-side and assembled with `ptxas -arch=sm_86 -O3`. No GPU involved.

| | `render` SASS | registers/thread | stack frame (bytes/thread) |
|---|---|---|---|
| before (`batch_invert_into`) | **27,016** | **255** | **20,896** |
| after (`batch_invert_stride2_into`) | **28,416** | **168** | **15,936** |
| delta | +1,400 (+5.2%) | −87 | −4,960 (−23.7%) |

Caveat, stated because it matters for reading that +5.2%: **`render`'s static
instruction count is a poor proxy for its dynamic cost here**, because ptxas
unrolls these loops. The same +1,400 appears at every `CHUNK` from 4 to 512
(§6), which is what an unroll artefact looks like, not what a per-element cost
change looks like. The honest dynamic measure is the multiply count, which is
exact and source-level (§4).

Supporting counts over the whole `render` function:

| `CHUNK` | variant | `render` SASS | `LDL` | `STL` |
|---|---|---|---|---|
| 16 | classic | 27,008 | 310 | 1,179 |
| 16 | stride-2 | 28,384 | 660 | 1,488 |
| 64 | classic | 27,488 | 310 | 1,659 |
| 64 | stride-2 | 28,744 | 660 | 1,848 |
| 256 | classic | 27,016 | 310 | 1,179 |
| 256 | stride-2 | 28,416 | 660 | 1,528 |

Static `LDL`/`STL` are also unroll-confounded (they do not scale with `CHUNK`
— the loops are unrolled and the unroll factor changes), so they are reported
as shape indicators only. The per-element traffic model in §3 is the one to
reason with.

---

## 3. Local memory traffic, per element

Exact, from the source. One `Fe` access = 5 × 64-bit words = 1 `LDL`/`STL`
instruction per word.

| | multiplies | `Fe` loads | `Fe` stores | bytes touched |
|---|---|---|---|---|
| classic | `3n` | `3n` | `2n` | `200n` |
| stride-2 | `3.5n` | `2.992n` | `1.496n` | `179.5n` |

At `n = 256`: multiplies 768 → 894, `Fe` stores 512 → 383, bytes touched
51,200 → 45,960 per thread per chunk (−10.2%).

So stride-2 does reduce traffic, by about a tenth — the halved store count
partly pays for the extra `els[i-1]` loads the rebuilds need. It does **not**
preserve traffic, and it does not eliminate it.

Note the `n=1` case: the peel short-circuits with `els[0] = acc_inv` and costs
zero multiplies, one of which is the single inversion the whole scheme exists
to amortise. `P[-1] = 1` is a compile-time constant, never a stored value.

---

## 4. Field multiplies per chunk

Per chunk of `CHUNK = 256`, the Montgomery part only; the two point-addition
walks and the compressions are unchanged.

| | Montgomery multiplies | rest of chunk | chunk total | squarings |
|---|---|---|---|---|
| classic | 768 (`3n`) | 4,093 muls | **4,861 muls** | 254 |
| stride-2 | 894 (`3.5n`) | 4,093 muls | **4,987 muls** | 254 |
| delta | **+126 (+2.59% of the chunk's multiplies)** | — | +126 | 0 |

"rest of chunk" = 2 walks × 255 `ge_madd` × 7 muls = 3,570, plus 256
compressions × 2 muls = 512, plus `fe_invert`'s 11 multiplies. One inversion
either way.

Closed forms (asserted in `stride2_multiply_counts_match_their_closed_forms`):
`7n/2 − 2` for even `n`, `(7n−5)/2` for odd `n`, `0` for `n ≤ 1`; against
`3n − 3` and `0` for the classic scheme.

### The exchange rate

Measured directly, by rebuilding `render` with dead `fe_mul`s in the peel and
differencing the static count:

    0 extra muls : 28,416
    1 extra mul   : 28,600   (+184)
    2 extra muls  : 28,776   (+360, i.e. +176/mul)
    4 extra muls  : 29,112   (+696, i.e. +174/mul)

**One `fe_mul` ≈ 175 SASS instructions.** A 64-bit `LDL`/`STL` is **1**
instruction. So the trade at `CHUNK=256` is:

    cost     : 126 extra muls x 175 SASS/mul   = +22,050 SASS per chunk
    saving   : 129 fewer Fe stores x 5 words    = -645 SASS per chunk
    net      : +21,405 SASS per chunk

i.e. **roughly 33 SASS spent per SASS saved.** Under the stated cost model
(total SASS instruction count, which per `CONTEXT.md` has been beating
occupancy gains on this kernel) that is a large, unambiguous loss.

The multiplies are on the critical path in a way the removed stores are not:
the rebuild `fe_mul` sits *inside* the peel's serial dependency chain
(`acc_inv` is carried from step to step), whereas the removed stores are
fire-and-forget. If anything this understates the cost.

---

## 5. The registers result, and why it does not rescue the trade

`ptxas` drops 255 → 168 registers. Decomposed against the base commit:

Isolating peel shape from buffer size, four cells:

| cell | peel shape | buffer | registers | stack frame |
|---|---|---|---|---|
| A | stride-2 (parity test, `i==0` early exit) | **half** | **168** | 15,936 |
| B | stride-2 (parity test, `i==0` early exit) | **full** | **168** | 21,056 |
| C | classic (`for i in (0..n).rev()`, no early exit) | full | 255 | 20,896 |
| D | classic, as C | half (out of bounds; probe only) | 255 | 10,752 |

Two things follow, and they point in opposite directions:

1. **The register drop comes from the peel shape, not the buffer.** Cells A and
   B have the same code and differ only in buffer size, and both get 168.
   Cells C and D have the classic code and both get 255, at either buffer size.
   So the halved `scratch` is *not* what buys the registers — reshaping the
   peel is, and it does so on its own.
2. **The 255 → 168 drop is real and reproducible.** 255 is the hardware
   ceiling and ptxas will happily sit on it; a marginally different loop shape
   moves it 34% off. At 168, occupancy would go from 8 to 12 warps/SM *if the
   host picked 128 threads per block*; at 256 threads per block (what `BLOCK`
   selects, and what the host's `geometry()` picks, since `by_occupancy =
   65536/168 = 390` and `min(256, 390) = 256`) it is 1 block/SM either way and
   occupancy is **8 warps/SM in both cases**.

So the register drop still buys nothing at the current launch geometry. But
unlike what I first assumed, it is *separable* from the extra multiplies: cell
B has the 168 registers with the full buffer and the stride-2 code, and a
follow-up should check whether some peel shape reaches 168 with the classic
`3n−3` multiply count. If so it is a free 1.5× occupancy at 128
threads/block, and it is the one lead here with real upside.

---

## 6. At what `CHUNK` does this stop being favourable?

**It never does.** The extra cost and the saving both scale linearly in `n`:

* extra multiplies: `+n/2` per chunk, i.e. `+0.5` per candidate;
* saved `Fe` stores: `−n/2` per chunk, i.e. `−0.5` per candidate.

The ratio is `n`-independent, so there is no crossover to find. The
per-element trade is fixed at **+87.5 SASS for −2.5 SASS**, a 35:1 loss, and
that holds for every `CHUNK` from 1 to 512.

Measured frame sizes confirm the buffer really does halve, and confirm it is
never the binding constraint:

| `CHUNK` | classic frame | stride-2 frame | delta |
|---|---|---|---|
| 4 | 1,216 | 1,296 | +80 |
| 8 | 1,376 | 1,376 | 0 |
| 16 | 1,696 | 1,536 | −160 |
| 32 | 2,976 | 2,496 | −480 |
| 64 | 5,536 | 4,416 | −1,120 |
| 128 | 10,656 | 8,256 | −2,400 |
| 256 | 20,896 | 15,936 | −4,960 |
| 512 | 41,376 | 31,296 | −10,080 |

Below `CHUNK=8` the arrays are small enough to be promoted out of local memory
entirely, so the change does nothing at all. Above it, the frame grows in both
cases and nothing in the launch geometry changes: occupancy is register-bound
at 1 block/SM regardless, and local memory per thread is not an occupancy
limiter on sm_86 at these sizes (the limit that binds is the 64K register file
at 255 registers).

The one regime where halving would pay is a *much* larger `CHUNK` — where
20 KB/thread of local memory starts to matter for something other than
occupancy. `CONTEXT.md` records that `CHUNK=512` already "ties on speed for
double the memory", which says the memory is not yet binding. The
regression/inversion-count trade mentioned in the brief
(`scratch: [Fe; SMALL]` plus sub-block looping) is strictly worse still: it
raises the inversion count proportionally, and an inversion is ~254 squarings
against a multiply's 25 products.

---

## 7. What would be worth doing instead

Ranked by measured payoff:

1. **Chase the 168-register shape without the extra multiplies.** §5 shows
   the register drop follows the peel's code shape, independent of buffer
   size: the stride-2 peel reaches 168 registers with a *full* buffer too,
   while the classic peel sits at 255 with either buffer. So the open question
   is whether some peel shape reaches 168 registers at the classic `3n−3`
   multiply count. If one does, it is a free 1.5× occupancy at 128
   threads/block. This is the only lead here with real upside.
2. **Leave `scratch` alone.** It is `2n−1` stored field elements for `3n−3`
   multiplies and one inversion, which is the correct shape for Montgomery's
   trick, and `3n−3` is optimal for a single inversion.
3. **Attack the double walk instead.** Per `CONTEXT.md` that is the dominant
   cost (~2 point additions per candidate). The inversion is amortised to
   `1/CHUNK` of a key; the walks are not amortised at all. Anything that
   removes a walk beats anything that removes `scratch`.

---

## 8. Correctness

`cargo +nightly test --manifest-path core/Cargo.toml --release` — **42 passed,
0 failed** (36 pre-existing + 6 new).

New tests, all in `core/src/group.rs`:

* `batch_invert_lengths_0_to_300` — both schemes against individual
  `fe_invert` for **every** length 0…300, which covers the odd-length tail,
  `P[-1] = 1`, and the lengths where the last prefix would or would not be
  stored.
* `batch_invert_degenerate_lengths` — **n = 0, 1, 2, 3** spelled out
  individually, since `n=1` is the case the peel can get structurally wrong
  and `n=2` is the smallest with a real peel.
* `batch_invert_large_lengths` — **prime-ish and awkward** lengths:
  7, 8, 9, 15, 16, 17, 127, 128, 129, 251, 255, 256, 257, 511, 512, 1023.
  251 and 257 are prime; 256 is a power of two (worst case for any halving
  scheme); 1023 = `2^10−1` so every element but the last lands in a full pair.
* `stride2_scratch_is_half_and_agrees` — pins the buffer contract
  (`stride2_scratch_len(n) == (n+1)/2`) so a future edit that needs the full
  length fails here rather than scribbling past the array, and checks the two
  schemes agree bit for bit.
* `batch_invert_tolerates_repeated_elements` — duplicates and adversarial
  values (`p−1`, maximal limbs, `18`). Montgomery's trick is input-agnostic,
  so repeated values are a cheap way to catch an indexing bug that distinct
  values happen to mask: several peel steps then consume the same slot.
* `stride2_multiply_counts_match_their_closed_forms` — the multiply counts in
  the table above are asserted, not asserted-in-prose, so the `SCRATCH.md`
  arithmetic cannot silently drift from the code.

The batch-inversion tests are checked against individual `fe_invert`, not
against the other implementation, so the two cannot be wrong in the same way.

### Mutation testing

Thirteen mutations of the new code, each of which must make `cargo test` fail.
All 13 are caught (`/tmp/opencode/wt/mutate.sh`):

```
CAUGHT   off-by-one in peel: i=0 short-circuit removed
CAUGHT   off-by-one in peel: acc updated after store (z[i] destroyed)
CAUGHT   off-by-one in peel: peel starts at n-2
CAUGHT   off-by-one in peel: peel starts at n
CAUGHT   parity flip: rebuild taken on even i instead of odd
CAUGHT   index slip: scratch[i/2 - 1] instead of scratch[i/2]
CAUGHT   index slip: scratch[i/2] -> scratch[i/2 + 1] in the even branch
CAUGHT   forward pass stores P[i] instead of P[i-1] (post-multiply)
CAUGHT   forward pass odd tail drops its store
CAUGHT   forward pass forgets the first element
CAUGHT   inversion replaced by identity
CAUGHT   n == 1 special case removed (falls into the peel)
CAUGHT   stride2_scratch_len returns n instead of (n+1)/2
```

The second one is the trap this change introduces and the classic scheme does
not have: in the peel, `acc_inv` must be advanced with the *input* `z[i]`, so
the output has to land in a temporary first. Writing it through `els[i]` first
feeds the reciprocal of `z[i]` back into the accumulator. The first mutation is
the one the brief warned about — the classic off-by-N in the reverse peel.

---

## 9. Recommendation

Do not ship this for speed. The change is committed on
`agent/nocombscratch` because it is correct, tested, and it does what it
claims — `scratch` halves, 10,240 → 5,120 bytes/thread — but it costs +2.59% of
the chunk's multiplies (+126 per chunk) to save 129 `Fe` stores, and at the
measured 175 SASS per `fe_mul` that is a ~33:1 loss on the cost model this
kernel is tuned for. The saving is also the wrong quantity: local-memory
capacity is not the binding constraint at `CHUNK=256` (occupancy is
register-bound at 8 warps/SM, unchanged by this patch), and the reduction does
not make the change favourable at any `CHUNK`, because both terms scale as `n`.

The reusable finding is §7.1: the 255 → 168 register drop is real, it is
attributable to the peel's loop shape rather than to buffer size, and it is
worth chasing for itself.
