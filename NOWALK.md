# Removing the second walk of each chunk

## What changed

`render` used to walk every chunk of `CHUNK` candidates twice:

```
pass 1: CHUNK-1 point additions, collecting each point's Z into zs[]
batch_invert_into(&mut zs, &mut scratch)   // one inversion serves the chunk
pass 2: RE-WALK the same CHUNK points (CHUNK more point additions),
        compressing each and testing it against the prefix
```

The re-walk existed because the compressed pubkey needs `Z^-1`, the batch
inversion cannot produce any of them until it has seen every `Z`, and the
`x = X/Z` and `y = Y/Z` divides therefore could not be done during the walk.

Two observations remove it:

1. **The `X/Z` divide is never needed.** A Tor onion address is
   `base32(pubkey || checksum || version)`, so the leading characters come from
   the *low* bits of the little-endian `y` coordinate. The sign of `x` lives in
   bit 255 of the encoding, and a 12-character prefix -- the longest the host
   will accept -- reaches only bit 59. So prefix testing needs `y` and nothing
   else.

2. **With only `Y/Z` to compute, `Y` is all that has to survive the inversion.**
   Keeping it costs 5 u64 per candidate of local memory and saves a whole point
   addition. `ge_madd` is 7 field multiplies plus 5 add/sub chains.

So pass 1 now stores `Y` alongside `Z`, and pass 2 becomes one `fe_mul`, one
reduction, and a 64-bit compare per candidate:

```
pass 1: CHUNK point additions, storing each point's Y and Z
batch_invert_into(&mut zs, &mut scratch)   // zs[c] becomes Z^-1
pass 2: for each c: y = Y[c] * Z^-1[c]; test the low 8 bytes of y
```

`field::fe_to_bytes_low8` is the new bit-level primitive. It returns the first
eight bytes of `fe_to_bytes(a)` packed little-endian, and it is exact rather
than approximate because of the structure of the packing: `fe_to_bytes` packs
five 51-bit limbs into one contiguous little-endian bit stream, and limbs 1..=4
contribute only at bit 51 and above. So once the value is reduced to its
canonical representative in `[0, p)` -- every limb a true digit below `2^51` --
the low 64 bits are exactly `t[0] | (t[1] << 51)`. The reduction is not
optional: a carry out of limb 0, or the subtraction of `p`, moves the low bits
while leaving the limbs above untouched. That is why this is stated in terms of
`fe_reduce` and tested against the same adversarial operand table that catches
`fe_reduce` bugs.

`BytePrefix` gains `prefix_word: u64`, the first eight prefix bytes packed
little-endian, and a `matches_low8(&self, w: u64)` that is the same predicate
as `matches(&[u8])` in register arithmetic: a masked compare of everything below
`last_byte_idx`, then a masked compare of the byte at it. The mask is applied to
the data side only, exactly as `matches` does, so a prefix byte with junk in the
bits the mask clears behaves as it did before. `matches` itself is unchanged and
still tested.

**Point 3 of the brief, verified rather than assumed.** The 64-byte expanded key
handed back on a match is `self.expanded`, the clamped scalar plus the nonce
prefix, and the compression path never writes to it -- the only writer is
`add_clamped_scalar`. The new code advances the scalar once per candidate in
pass 2, in the same order relative to the tests as the old `state.advance()` did
after each `check`. `expanded_key_is_independent_of_the_pubkey` asserts the
chunked walk's sequence of expanded keys is element-for-element equal to the
sequential walk's, and `chunked_walk_matches_sequential_walk` checks the same
pairwise against an independent per-candidate inversion while also checking the
full compressed pubkey agrees. Both confirm the claim.

## Static quantification

All numbers from `ptxas -arch=sm_86 -O3` on the PTX that `build.rs` feeds the
host, and from `cuobjdump -sass` / `-res-usage` on the resulting cubin. No GPU
was touched. Baseline is commit `b0a8695` built from a pristine `git archive` of
the same tree, so the two builds differ only in this change. Repeated builds are
bit-identical, so these are not build noise.

### `render` instruction count, whole function

| | base `b0a8695` | nowalk | delta |
|---|---|---|---|
| PTX instructions | 9,962 | 7,883 | **-2,079 (-20.9%)** |
| SASS instructions | 27,016 | 23,128 | **-3,888 (-14.4%)** |
| SASS `IMAD.WIDE.U32*` (multiply pipe) | 5,598 | 4,751 | **-847 (-15.1%)** |

`IMAD.WIDE` is the right multiplier proxy: per the cost model in `CONTEXT.md` a
64x64->128 multiply is 4 IMAD-class ops, and a `fe_mul` is 25 of them, so 100
`IMAD.WIDE` is one field multiply. The 847 removed is **8.47 field multiplies
per candidate**, which is the 7 of the removed `ge_madd` plus the 1 of the
removed `X/Z` divide, plus 0.47 of incidental carry/reduction work that came out
of dropping `fe_to_bytes`'s 32-byte packing loop.

### Per-candidate cost, measured as a marginal

The whole-function count mixes per-launch fixed cost (`begin`: SHA-512 plus the
63-addition comb) with per-candidate work. To isolate the latter, the kernel was
rebuilt at `CHUNK = BATCH = 1, 2, 4` so the loops unroll, and the difference
between successive sizes taken:

| | base | nowalk | saving |
|---|---|---|---|
| marginal SASS / candidate | 2,256 | 1,312 | **-944 (-41.8%)** |
| marginal `IMAD.WIDE` / candidate | 542 | 314 | **-228 (-42.1%)** |

The 228 `IMAD.WIDE` is 2.28 field multiplies per candidate, a smaller figure
than the whole-function 8.47 because at `CHUNK <= 4` the batch inversion and
its scratch are a much larger fixed share of each candidate's cost and the
compiler has not yet hoisted the loop structure it uses at `CHUNK = 256`. The
whole-function number at the shipping configuration is the one to trust.

### Hot loop, isolated at `CHUNK = 256`

The chunk loop is the largest backward branch in the function. Its body:

| | base | nowalk | delta |
|---|---|---|---|
| pass-1 body (walk + store) | 3,485 SASS, 745 `IMAD.WIDE` | 3,487 SASS, 745 `IMAD.WIDE` | +2, 0 |
| pass-2 body (compress + test) | 8,088 SASS, 1,700 `IMAD.WIDE` | 4,158 SASS, 853 `IMAD.WIDE` | **-3,930 (-48.6%)** |
| whole chunk (both passes + inversion) | 17.00 M | 8.53 M | **-8.47 M** |

Pass 1 is unchanged to within two instructions, which is the point: the walk
still costs exactly one `ge_madd` per candidate. Pass 2 drops by almost half
because it no longer contains a point addition. The `STL.U8` in pass 2 falls
from 41 to 3, which is `fe_to_bytes`'s 32-byte packing loop -- the encoding is
never materialised as bytes any more.

### Registers and local memory

| | base | nowalk |
|---|---|---|
| registers/thread | 255 (hardware ceiling) | **168** |
| local bytes/thread (`STACK`) | 20,896 | **31,104** |
| local bytes/thread by hand | 2 x 10,240 + 416 | 3 x 10,240 + 384 |
| warps/SM by registers | 8 of 48 | **12 of 48** |

The local memory delta is +10,208 B/thread, i.e. the `ys` array (10,240 B at
`CHUNK = 256`) less 32 B of incidental shrink. This is the trade the brief
anticipated, and it is the same shape CONTEXT.md describes for the `CHUNK=256`
retune: `STACK` grows, registers *fall*, occupancy rises.

The register fall is the interesting part, and it was not the expected sign.
`ptxas` needed 255 before because pass 2 held a live `Ge` across the inversion
and a 32-byte key array across the compare; with neither, the live set across
`batch_invert_into` is one `Fe`, and the allocator has room. At 168 registers
the register file admits 12 warps/SM against 8, a **+50% occupancy** change on
top of the -14.4% instruction count. Both metrics moved the right way, so
occupancy is not being traded away for instructions here.

Caveat on the register figure: it is `ptxas`'s allocation at `-arch=sm_86`, not
what the driver JITs. The driver re-runs `ptxas` on the PTX at load with its own
defaults and the host reports that number under `--verbose`, so the two may
differ. The direction is robust (the live set genuinely shrank); the exact 168
should be confirmed by `--verbose` on the real device.

### CHUNK/BATCH sensitivity

Because the memory cost per candidate went up by 50% and the register pressure
went down, the old `CHUNK = 256` optimum may have moved. Statically:

| CHUNK/BATCH | base SASS | nowalk SASS | base regs | nowalk regs | nowalk local |
|---|---|---|---|---|---|
| 128/512 | 27,024 | 23,136 | 255 | 255 | 15,744 |
| 256/512 | 27,016 | 23,128 | 255 | **168** | 31,104 |
| 256/1024 | 27,024 | 23,136 | 255 | 255 | 31,104 |
| 512/512 | 26,952 | 23,072 | 255 | 255 | 61,824 |

`CHUNK = 256, BATCH = 512` is the only cell where the compiler finds the 168
register allocation, so it should stay the default. `CHUNK = 128` is worth an
A/B on the real device: half the memory, and at 255 registers it is no worse
than the baseline's occupancy. The instruction counts barely move with `CHUNK`
(within 0.3%), which says the sweep is really about memory and registers.

## Risks and what I could not resolve

**Unmeasured.** Every number above is static. The brief's hard constraint ruled
out touching the GPU, so there is no keys/s for this change, and given the
measurement discipline in CONTEXT.md (sequential A/B is worthless, the first run
after a build is 20-30% low) an unmeasured guess would not be worth much
anyway. What can be said: -14.4% instructions and -15.1% on the multiply pipe,
with occupancy *rising* from 8 to 12 warps/SM, is a favourable direction on both
axes of the stated cost model, and the per-candidate field-multiply count drops
from ~16 to ~8. I expect a real win, but I have not measured one and the
`CHUNK = 128` memory question is unresolved.

**Local memory traffic.** `ys` and `zs` are both read and written once per
candidate, so local memory traffic per candidate roughly doubles (two 40-byte
arrays streamed instead of one). At 20 SMs and 12 warps that is real bandwidth,
and it is the main way this could disappoint: an occupancy gain that does not
translate into bandwidth is worth less than the instruction count suggests.
Whether it is 30 KB or 15 KB of live local state per thread is exactly what the
`CHUNK = 128` A/B would settle.

**A prefix longer than 12 characters is now rejected** by the host with an
explicit assertion rather than silently mis-tested. Previously any length
"worked" (and would simply never match, since no search could finish). This is
a behaviour change on an input that could not have succeeded. 12 characters is
60 bits, a 2^60-candidate search; at 149M keys/s that is 20,000 years.

**`matches_low8` fails closed** on a prefix needing more than eight bytes rather
than shifting by 64 or matching a truncated prefix. The host makes that
unreachable, and `matches_low8_rejects_a_prefix_wider_than_the_word` pins it, but
it is a new failure mode: a mis-built `KernelParams` now yields zero matches
rather than garbage matches. Zero matches is the right direction to fail.

**The `advance()` method is now used only by the no-prefix benchmark path** and
by tests; the matching path advances the scalar directly. `advance()` also bumps
`self.index`, and `index` is dead state (nothing reads it). I kept the field and
bumped it once per chunk to stay honest, but a reader may wonder why the two
paths treat it differently. It is a candidate for deletion, not part of this
change.

**Not attempted: storing `Y` in fewer bits.** Only the low 8 bytes of `y` are
needed, so in principle one could avoid materialising the full `Y` and compute
`Y * Z^-1` mod low bits somehow. There is no way to do that: the low bits of a
modular quotient depend on the whole dividend, so `Y` has to be kept whole. The
10 KB is the floor for this approach.

**Not attempted: dropping the second array instead of the second walk.** The
alternative to `ys` is to keep the re-walk and drop the memory, which is what
the base does. A third option exists and was not explored: since
`batch_invert_into` needs a `scratch` array anyway, and the backwards pass reads
`els[i]` after having overwritten it, a fused variant could invert in place with
`CHUNK` slots total rather than `2 * CHUNK` -- that would recover the 10 KB
*and* keep the single walk. It changes the multiply count of the inversion
(trade a multiply for a local read per element) and deserves its own
investigation.

## Files touched

- `core/src/field.rs` -- `fe_to_bytes_low8`, plus three tests
- `core/src/lib.rs` -- `BytePrefix::prefix_word`, `matches_low8`,
  `pack_prefix_word`, `MAX_PREFIX_CHARS`, plus four tests
- `core/src/kernel.rs` -- single-walk `render`, `check` takes `y`, the chunk
  walk rewritten in the tests so they exercise the same statements the device
  runs, plus three new tests
- `src/gpu.rs` -- `prefix_word` plumbing, the length assertion, one `debug_assert`

46 tests pass (was 36) at the default configuration and at `CHUNK`/`BATCH` of
1/1, 2/2, 3/3, 4/8, 8/16 and 16/32. `cargo build --release` is clean apart
from the pre-existing `unused dependency num_cpus` and `unused import BLOCK`
warnings, and `t3v --help` runs.
