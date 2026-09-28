# Comb parameters (`COMB_W`)

Analysis of the fixed-window comb's window width, done statically: arithmetic on
the generated code plus `ptxas -arch=sm_86` instruction counts. **No GPU was run.**
Every SASS number below is from a cubin in this worktree, and the recipe to
reproduce it is at the bottom.

Part of the `agent/norepack` branch, on top of `b0a8695`.

---

## 1. Recommendation

**Keep `COMB_W = 4`.** It is the only width that is simultaneously

- the cheapest in *instruction count* among the widths whose table fits in L1, and
- the largest width whose table still fits in L1, and
- already the measured default, so changing it is a risk with no measured upside.

The arithmetic behind each clause is below. The short version: going from W=4 to
W=3 saves 8% of the comb's instructions but the comb is only **5.8% of the
kernel** at `BATCH=512`, so the saving is ~0.5% of total SASS — inside the noise
floor of an A/B on this part. Going the other way, to W=6, saves another ~1.5%
of the comb (0.1% of the kernel) and costs a **2.5x** table that no longer fits
in L1, which is a much larger and much less predictable penalty. There is no
width that wins on both axes, and the default sits at the knee.

What this branch *does* change is that the reasoning is now enforced rather than
asserted, and the correctness of every width is now tested rather than assumed.

---

## 2. What the code does, and what changed

Before this branch, `COMB_W` was validated by two runtime `assert!`s — one in
`build_comb_table` on the host, one in `render` on the device. Both were
`assert!(COMB_W >= 2 && COMB_W <= 4)` and `assert!(252 % COMB_W == 0)`.

Neither is a real guard. `COMB_W` is a `const`, so both conditions are
compile-time constants, so LLVM folds them away. Verified:

```
$ T3V_COMB_W=5 cargo build --release          # before this change
   ...succeeds, and produces a 73-line PTX:
   .visible .entry render(...) { call.uni panic; exit; }
```

A 4-instruction kernel that traps on launch. The failure surfaces when someone
runs the binary, on the GPU, as a launch failure — not as a build error. And the
host-side `assert!` in `build_comb_table` fires inside a per-device worker
thread, so it surfaces as a panic on a detached thread while the process
carries on.

Now the three preconditions are `const _: () = assert!(...)` in
`core/src/group.rs`, so a bad `T3V_COMB_W` is a **build error** with a message
that names the rule it broke:

```
$ T3V_COMB_W=5 cargo build --release
error[E0080]: evaluation panicked: T3V_COMB_W: the comb window must divide
COMB_BITS (252), or the top windows are silently dropped -- 252 % 5 == 2 drops
bits 253 and 254, and clamping always sets bit 254
error[E0080]: evaluation panicked: T3V_COMB_W: the comb table must fit in the
128 KiB L1, or all eight warps re-stream it every launch; W=5 already needs 181 KiB
```

Verified for `T3V_COMB_W` in {1, 5, 7, 8, 9}; all five are build errors, and
2, 3 and 4 still build.

The second change is that the comb is now **generic over `W`**
(`comb_slot_w::<W>`, `scalarmult_base_comb_w::<W>`, `build_comb_table_w::<W>`)
with `COMB_W` only an instantiation. This is a testability change, not a
performance one — see §7 for the register and instruction deltas, which are
zero at W=4 apart from three `IMAD`s in a 3456-instruction loop body.

---

## 3. Candidate widths, and the remainder

`COMB_BITS = 252` is the count of scalar bits in play: clamping clears bits
0, 1, 2 and bit 255, and forces bit 254, so bits 3..=254 inclusive survive and
`252 = 4 * 63 = 2^2 * 3^2 * 7`. `COMB_WINDOWS = COMB_BITS / COMB_W` is
**integer division**, so a width that does not divide 252 does not error — it
truncates, and the top `252 mod W` bits are *never looked at by anyone*.

| W | divides 252? | windows | top bit covered |
|---|---|---|---|
| 2 | yes (126) | 126 | 254 |
| 3 | yes (84) | 84 | 254 |
| 4 | yes (63) | 63 | 254 |
| 5 | **no**, 252 = 50*5 + 2 | 50 | **252** — bits 253, 254 dropped |
| 6 | yes (42) | 42 | 254 |
| 7 | yes (36) | 36 | 254 |
| 8 | **no**, 252 = 31*8 + 4 | 31 | **252** — bits 253, 254 dropped |
| 9 | yes (28) | 28 | 254 |

W=5 is the interesting failure, because it is *plausible*. `252 % 5 == 2`, so
50 windows of 5 bits cover bits 3..=252 and bits 253 and 254 are simply not
part of the scalar multiplication. Bit 254 is set by clamping on **every
scalar**, so the W=5 comb is wrong on 100% of inputs, not on a rare tail.

This is now demonstrated rather than asserted. `w5_would_drop_the_top_bits`
builds the W=5 table with the deliberately unvalidated `build_comb_table_at`
and checks two things:

1. it does **not** equal the power-table ladder, and
2. it **does** equal the ladder over the scalar with bits 253 and 254 cleared.

The second is what makes it a dropped-bit bug rather than a coincidence.

`w5_would_drop_the_top_bits` is the reason `build_comb_table_at` exists and
carries a `# Performs no validation` contract. Everything in the crate goes
through `build_comb_table_w`, which gates on `comb_w_ok`.

### The digit-extraction bound is 9, not "whatever"

`COMB_W_MAX` was not previously stated as a number at all; the old comment said
`COMB_W <= 5`, which is why the assert read `2..=4`. The real bound is derived
in `comb_slot_w`: a window is `W` bits at a bit offset of at most 7, read out of
a two-byte value covering bits 0..=15, so `off + W - 1 <= 15` and the worst
case `off == 7` gives **`W <= 9`**. At W=10 the extractor would silently drop
the top digit bit and return a slot from the wrong half of the window — the
same class of bug as W=5, and just as silent.

W=9 is arithmetically fine (28 windows of 512 slots) and is rejected by the L1
rule instead, at 1.6 MiB. `comb_slot_read_is_safe` checks the general
per-window condition for every `(W, window)` pair in `2..=9` rather than
trusting the derivation to survive a change to `COMB_FIRST_BIT`.

---

## 4. Table size and gather cost per W

Table entry is a `Niels` = 3 field elements = **120 bytes**. `slots = 2^W`
(the identity is slot 0, so the window loop stays branchless).

### 4a. Table size

| W | windows | slots/window | entries | table bytes | × 128 KiB L1 | crosses L2 per SM per launch |
|---|---|---|---|---|---|---|
| 2 | 126 | 4 | 504 | 60,600 (+120 for `8B`) = **60,720** | 46.3% | 60,720 |
| 3 | 84 | 8 | 672 | 80,760 (+120) = **80,880** | 61.7% | 80,880 |
| 4 | 63 | 16 | 1,008 | 120,960 (+120) = **121,080** | 92.4% | 121,080 |
| 5 | 50 | 32 | 1,600 | 192,000 (+120) | **147%** | 1,536,000 |
| 6 | 42 | 64 | 2,688 | 322,560 (+120) | **246%** | 2,580,480 |
| 7 | 36 | 128 | 4,608 | 552,960 (+120) | **422%** | 4,423,680 |

The last column is the one that matters and it is not a smooth curve. At
255 registers/thread only **one 256-thread block is resident per SM**, so there
are 8 warps per SM and all of them run the comb once per launch, marching
through the same 63 (or `n`) window blocks in roughly the same order. If the
table fits in L1, the first warp pulls it in and the other seven hit: it crosses
L2 **once** per SM per launch. If it does not fit, each warp re-streams it and
it crosses L2 **eight times** — an 8x cliff, not a gradual slide.

That cliff lands between W=4 and W=5. W=4 sits at 92.4% of L1, which is tight
but resident, and it is the *largest* width for which that is true.

### 4b. Gather cost

A comb lookup is not a broadcast. `j` is a loop counter and is warp-uniform, but
the *digit* is per-lane, so the 32 lanes read 32 different addresses inside one
window block of `2^W * 120` bytes. L1 services a request in 128-byte lines, so
one `LDG.E.64` costs one line-fill per distinct 128-byte line the lanes touch.

| W | window block | L1 lines/block | distinct slots a warp can hit | line-fills per lookup | L1 bytes per scalarmult |
|---|---|---|---|---|---|
| 2 | 480 B | 3.8 | 4 | 3.8 | 59.1 KiB |
| 3 | 960 B | 7.5 | 8 | 7.5 | 78.8 KiB |
| 4 | 1,920 B | 15.0 | 16 | 15.0 | 118.1 KiB |
| 6 | 7,680 B | 60.0 | 32 (capped by warp width) | 30.0 | 157.5 KiB |
| 7 | 15,360 B | 120.0 | 32 | 30.0 | 135.0 KiB |

Two things fall out of this table that the "window block" column alone hides:

- **Per-lookup line-fills saturate at 32** — the warp width. Beyond W=5 there is
  no more waste to trim, because all 32 lanes are already spread across as many
  lines as a warp can address. Going W=6 -> W=7 makes each *lookup* cheaper
  (30 -> 30 line-fills, same) while the *number* of lookups falls 42 -> 36, so
  total traffic does drop. The curve is not monotonic; it has a turning point.
- **The useful payload is 120 bytes of a 128-byte line**, so the gather wastes at
  most 6% on transfer granularity. The waste is not in the bytes, it is in the
  *count of separate line-fills*, and that is what a shared-memory table would
  remove (a shared-memory gather has no line granularity at all).

At 118.1 KiB per scalar multiplication and one comb per 512 candidates, that is
**236 bytes of L1 traffic per candidate**. For scale, the walk's own per-candidate
local-memory traffic is the same order: `batch_invert_into` reads and writes each
field element twice (once in the forward prefix pass, once in the backward
peel), so 160 B/candidate, plus 40 B stored to `zs[c]` and 40 B read back =
**240 B/candidate of `LDL`/`STL`**. So the comb's entire gather costs about as
much L1 traffic as the chunk buffers the kernel has no choice about — and both
are dwarfed by the 3,456-instruction arithmetic that issued them.

The comb's lookup is not a bottleneck at any usable W. It only becomes one if
the table stops fitting.

---

## 5. `render` SASS and register counts

`ptxas -arch=sm_86 -O3 -v`, one cubin per width, from this worktree.

| W | registers | stack frame | spills | static SASS (whole kernel) | comb window loop body |
|---|---|---|---|---|---|
| 2 | 255 | 20,896 B | 0 | 27,024 | 3,456 |
| 3 | 255 | 20,896 B | 0 | 27,024 | 3,458 |
| 4 | 255 | 20,896 B | 0 | 27,024 | 3,456 |

Identical on all three, and identical to the pre-change baseline at W=4: 27,016
static instructions before, 27,024 after, same 255 registers, same 20,896 B
stack frame, still 0 spills. The +8 is 3 `IMAD`s in the comb window loop body
(3453 -> 3456, +0.09%) and 5 `NOP`s elsewhere; it is loop-scheduling noise from
the const-generic instantiation, not a real cost. The `assert!`s removed from
`render` were already dead (see §2), so removing them cost nothing.

**Register count does not depend on W at all**, which is the important negative
result. The comb is a `for` loop over `COMB_WINDOWS`; `ptxas` does not unroll it
(the loop body contains 15 `BSSY`/`BSYNC` pairs and 3,456 instructions, far past
any unroll heuristic). So W changes the loop *trip count*, not the live-state
footprint. Occupancy stays at 255 registers -> 8 warps/SM of 48 for every
candidate width, and the register ceiling is not a reason to prefer a narrower
window.

The comb window loop body, at W=4, decomposes as:

| | count |
|---|---|
| total instructions | 3,456 |
| of which `IMAD` | 1,399 |
| `LDG.E.64` (the `Niels`, 15 u64 = 120 B) | 15 |
| `LDL.U8` (the two digit-extraction byte reads) | 2 |
| `BSSY` (carry-chain loop boundaries in `fe_mul`) | 15 |
| **a bare `ge_madd` with the addend already in registers** | **3,373** |

So the table lookup itself costs **83 instructions per window** — 15 `LDG.E.64`
plus address arithmetic and 2 local byte loads — on top of the 3,373-instruction
addition. The lookup is 2.4% of a window. Reducing it (say, by keeping the
current window's `Niels` in registers across two windows) could save at most
2.4% of the comb, which is 0.14% of the kernel. Not worth the complexity.

### The first window is already nearly free

This is the one place where a special case *does* pay, and the compiler already
takes it. `ge_madd` is called against `ge_identity()`, a compile-time constant,
so LLVM peels window 0 out of the loop and constant-folds it: the three
multiplies against the identity's known `1, 1, 0` collapse.

Measured in the PTX:

| region | `mul` instructions | `ld.local.b8` |
|---|---|---|
| peeled first window | **55** | 33 |
| loop body (windows 1..62) | **375** | 2 |

The peel's digit extraction is a single `bfe.u32 %r76, %r75, 3, 4` — a fused
bit-field extract, no second byte load, no shift, no mask — because window 0's
bits 3..6 are all in byte 0 with no wrap. The loop body needs the full
two-byte read, the shift and the mask.

So window 0 costs about **one field multiply** instead of seven, and the comb is
really `62 * 7 + 1 = 435` multiplies per scalar multiplication, not `63 * 7 =
441`. The `ge_madd`-against-identity code in `scalarmult_base_comb_w` has a
comment recording this so nobody "simplifies" it into something that loses the
peel.

---

## 6. Instruction cost per W, and the crossover

### 6a. Cost per scalar multiplication

Using the measured 3,456 instructions per window and the peel discount:

| W | windows | additions | 7M-equivalents | SASS per scalarmult | vs W=4 |
|---|---|---|---|---|---|
| 2 | 126 | 126 | 881 | 441,819 | **+92.7%** |
| 3 | 84 | 84 | 587 | 300,153 | **+30.9%** |
| 4 | 63 | 63 | 440 | 229,320 | baseline |
| 6 | 42 | 42 | 293 | 158,487 | **-30.9%** |
| 7 | 36 | 36 | 251 | 138,249 | **-39.7%** |

The 7M column is the classic all-subsets comb count and it is the *right* metric
to reason with, because it says what happens on hardware with a native 64-bit
multiply. On this part a 7M addition is 3,373 SASS instructions, not 7
operations, and the same ratio holds: instruction count is the cost model.

### 6b. Amortised over BATCH

Per-candidate cost, from the measured loop bodies (pass-1 walk 3,373, pass-2
walk + compress 3,476, batch-invert 3,475 amortised over CHUNK=256):

| BATCH | per candidate | comb per candidate | **comb share of kernel** |
|---|---|---|---|
| 64 | 6,863 | 3,402 | 33.1% |
| 128 | 6,863 | 1,701 | 19.9% |
| 256 | 6,863 | 850 | 11.0% |
| **512** | **6,863** | **425** | **5.8%** |
| 1024 | 6,863 | 213 | 3.0% |
| 2048 | 6,863 | 106 | 1.5% |
| 4096 | 6,863 | 53 | 0.8% |

**This is the answer to "at what point does the comb stop being the
bottleneck": it already has.** At the current `BATCH=512` the comb is 5.8% of
the kernel's instructions. It was 33% at `BATCH=64`, which is why the CHUNK/BATCH
retune (64/8 -> 512/256, +121%) helped so much more than the comb did: the
retune cut the comb's share by 5.7x while leaving the comb itself untouched.

The comb never becomes *free*, but at 5.8% its width is a second-order
parameter. That is the whole reason the sweep was worth redoing now: at
`BATCH=64` the answer would have been "as small as possible", and at
`BATCH=512` the answer is "whatever fits in L1 and is cheapest".

### 6c. Would a cheaper table win on L1 hit rate?

This is the question the retune made interesting, and the answer is **no**,
because the L1 saving is worth less than the arithmetic it costs.

Going W=4 -> W=2 shrinks the table from 92.4% to 46.3% of L1. Both are
*resident* — neither thrashes — so the saving is not a cliff-avoidance, it is
just "more L1 left over for other things". And there is nothing else using L1
during the comb phase: the comb runs once at the very start of the kernel,
before any of the walk's local-memory traffic, and it is over long before the
`zs`/`scratch` arrays (20 KiB/thread of local memory) start cycling.

The price of that headroom is 63 extra window additions per batch:

| BATCH | extra SASS/candidate for W=2 vs W=4 | as % of a candidate |
|---|---|---|
| 512 | +425 | 6.2% |
| 1024 | +213 | 3.1% |
| 2048 | +106 | 1.6% |
| 4096 | +53 | 0.8% |

**Crossover: never, at any BATCH this kernel can run.** To make W=2's extra
arithmetic worth 1% of the kernel, the L1 headroom it buys would have to be
worth 1% of the kernel too — and it buys nothing, because W=4's table is already
resident. A narrower table only becomes attractive if the table *stops* being
resident, and that is a discontinuity at W=5, not a gradient from W=4 down.

So the two candidate directions both fail:

- **Narrower (W=3, W=2):** strictly more arithmetic, no L1 benefit, because the
  wide table already fits. W=3 saves 21 windows = 72,576 SASS per scalarmult,
  which at BATCH=512 is 142 SASS/candidate = **2.1% of a candidate's cost** and
  ~0.12% of the kernel. Below the A/B noise floor, and it costs 30 KiB of L1
  headroom that nothing wants.
- **Wider (W=6, W=7):** saves 21 and 27 windows respectively, worth 142 and 182
  SASS/candidate = 2.1% and 2.7% of a candidate, and the table jumps to 246% and
  422% of L1, which is the 8x re-stream cliff from §4a. The resulting L2
  traffic, at 9.5 launches/s and 20 SMs:

  | W | L2 B/SM/launch | device GB/s |
  |---|---|---|
  | 4 | 121,080 | 0.023 |
  | 6 | 2,580,480 | 0.49 |
  | 7 | 4,423,680 | 0.84 |

  So W=6 costs 21x the table traffic of W=4 and W=7 costs 37x, to save 2.1% and
  2.7% of a candidate. For reference the walk's own local-memory traffic is
  240 B/candidate × 149.4M keys/s = **36 GB/s** through L1, so a 0.5-0.8 GB/s
  addition is not fatal on its own — but it is 0.5% of the L1/L2 budget spent
  to buy 2% of arithmetic, in the one regime where the table is *known* not to
  be resident, and that asymmetry is not something to bet a measured 149.4M
  keys/s on.

**W=4 is the knee.** If it had to be justified by one number, it is: 63 is the
fewest windows of any width whose table still fits in L1.

### 6d. What would actually move the needle

Not `COMB_W`. In descending order of measured SASS weight:

1. **The double walk.** 2 × ~3,400 instructions per candidate, ~99% of the
   kernel's arithmetic. Eliminating one of the two passes (by buffering the
   points instead of re-walking, which costs local memory) is the only change
   with double-digit potential.
2. **`batch_invert_into`'s two muls per element.** 3,475 instructions per chunk
   / 256 candidates = 13.6 per candidate. Montgomery's trick is already at 3
   muls/element plus 1 inversion; there is a standard 2-mul variant using
   Montgomery's trick over *differences* that is ~30% cheaper on the batch part.
3. **The `Niels` load, 83 instructions per window** (2.4% of the comb, 0.14% of
   the kernel). A shared-memory comb table would make the lookup a
   `LDS.64` bank-conflict-free gather instead of a 15-line `LDG` sequence. Not
   worth it at 0.14%.
4. **`COMB_W`.** 5.8% of the kernel, and moving between the neighbouring legal
   values changes that by ±1-2% of the kernel. Last.

---

## 7. `COMB_W_MAX`: why 9 and not 4

The old assert said `2..=4`. The new one says `2..=9` and then adds the two
substantive rules, because conflating "what the arithmetic allows" with "what is
worth running" is what made the old bound both unenforced and under-explained.

- `COMB_W_MIN = 2` — W=1 is the `[2^i]B` ladder the comb replaced, so it is
  excluded by name rather than by arithmetic.
- `COMB_W_MAX = 9` — the extractor's two-byte read. At W=10 the top digit bit
  is silently dropped.
- `COMB_BITS % COMB_W == 0` — or the top windows vanish.
- `comb_table_bytes(COMB_W) <= COMB_L1_BYTES` — or the 8x re-stream cliff.

The last two are the ones with teeth, and both are now compile errors rather
than a device trap. `COMB_L1_BYTES = 128 * 1024` is sm_86's unified L1/shared
block, and it is the reason the check lives in `group.rs` next to the table
layout rather than in a comment in `01-comb.md`.

---

## 8. Tests

45 host-side tests (was 36), all passing:

```
cargo +nightly test --manifest-path core/Cargo.toml --release
```

| test | what it pins down |
|---|---|
| `comb_slot_reads_the_right_bits` | the extractor, at the built width (was: the only coverage) |
| `comb_slot_reads_the_right_bits_for_every_window` | the extractor at W=2,3,4,6,7, against a bit-at-a-time oracle, over 5 scalars including all-ones and bit-255-set |
| `comb_top_window_lands_on_bit_254` | the top window ends at bit 254, and its wrapped byte read is fully masked — separately, so a failure says which broke |
| `comb_slot_read_is_safe` | for every `(W, window)` in 2..=9: the high byte is either a real index or provably masked away, and the read never passes bit 15 |
| `comb_w_ok_matches_its_own_definition` | the usability predicate, pinned on both sides of each boundary |
| `w5_would_drop_the_top_bits` | **the W=5 bug, demonstrated**: builds the table, shows it disagrees with the ladder, and shows it equals the ladder over bits 3..=252 |
| `comb_window_count_is_exact` | windows × W == 252 and the byte arithmetic, for each accepted W |
| `comb_matches_power_ladder` | end-to-end vs the ladder at the built width |
| `comb_matches_power_ladder_for_every_window` | end-to-end at W=2,3,4,6,7, 64 pseudo-random clamped scalars each |
| `comb_rfc8032_vectors_for_every_window` | the published RFC 8032 pubkeys at W=2,3,4,6,7 |
| `comb_slots_are_subset_sums` | every slot of every window at W=2,3,4, rebuilt from the power table |
| `comb_slot_zero_is_the_identity_at_every_window` | slot 0 really is the identity at W=2,3,4,6,7 — the property that makes the window loop branchless |

W=6 and W=7 are tested even though `comb_w_ok` rejects them, on purpose: the
*maths* is right at those widths and only the table size is not. If a future
change makes a wide window cheap (a shared-memory table, say), the correctness
side is already covered and only the size rule would need revisiting.

---

## 9. Reproducing the SASS numbers

```sh
export CUDA_PATH=/opt/cuda
T3V_COMB_W=4 cargo +nightly build --release
ptxas -arch=sm_86 -O3 -v \
  core/target/nvptx64-nvidia-cuda/release/tor_v3_vanity_core.ptx -o /tmp/w4.cubin
cuobjdump -sass /tmp/w4.cubin > /tmp/w4.sass
```

The comb window loop is identifiable in the dump as the only loop whose body
contains exactly 15 `LDG.E.64` (the `Niels`) and 2 `LDL.U8` (the digit byte
reads), with a `@P0 BRA` back to its own top. At W=4 that is
`0x0ab10..0x18300`, 3,456 instructions. The bare `ge_madd` it contains — 3,373
instructions, no `LDG`, no `LDL` — is the pass-1 walk loop at `0x1a6f0..0x279b0`,
which is the cleaner reference for "what one addition costs here".

The peeled first window is visible in the PTX rather than the SASS, between
`%bb.25` and the comb loop header `$L__BB0_30`: 55 `mul` against the loop
body's 375, and a `bfe.u32` where the loop has two `ld.local.b8`.

**No GPU workload was run to produce any number in this document.** Every
figure is either closed-form arithmetic over the table layout, or a count of
instructions in a cubin built from this worktree.
