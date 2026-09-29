# t3v

Generate Tor v3 vanity `.onion` hostnames by brute-forcing ed25519 keys on an
NVIDIA GPU.

An onion address is `base32(pubkey || checksum || version)`, so its first
characters are a function of the public key. A key is derived from a clamped
scalar, and each scalar gives one key, so a prefix of *N* base32 characters is
found by generating keys until one matches -- 2^(5N) candidates in expectation.
The work is one point addition per candidate: the scalars that remain clamped
after clamping are exactly the multiples of 8, so the search walks an arithmetic
progression and every step yields a testable key.

## Install

Requires Rust nightly, the `nvptx64-nvidia-cuda` target, and CUDA.

```sh
rustup toolchain install nightly
rustup component add llvm-bitcode-linker --toolchain nightly
rustup target add nvptx64-nvidia-cuda --toolchain nightly
cargo +nightly install --path .
```

Nightly is not optional. The kernel crate uses `#![feature(abi_ptx,
stdarch_nvptx)]`, and there is no `rust-toolchain` file, so a bare `cargo build`
fails on the device crate.

If CUDA is not at `/usr/local/cuda` or `/opt/cuda`:

```sh
CUDA_PATH=/path/to/cuda cargo +nightly install --path .
```

## Use

```sh
# one key per file, in a directory
t3v --dst mykeys/ myprefix

# several prefixes at once
t3v --dst mykeys/ foo,bar,baz

# every match in one file, truncated on each run
t3v -o keys.csv foo
t3v -o keys.json --format json foo

# ...or add to it across runs, so a long search can be resumed
t3v -o keys.csv --append foo

# replacing an existing file has to be asked for by name
t3v -o keys.csv --overwrite foo

# stop once you have enough, rather than watching for them
t3v -n 10 foo

# just the addresses, nothing else, for piping
t3v --quiet foo | tee addrs.txt

# a key Tor will accept as hs_ed25519_secret_key
cat mykeys/foo*.onion > /var/lib/tor/hidden_service/hs_ed25519_secret_key
```

Prefixes are base32 (`a-z`, `2-7`), at most 12 characters, and case is ignored.
Output is one file per key by default; `--out` writes every match to a single
file, whose encoding follows `--format` or, failing that, the file extension.

Writing to an `--out` file that already exists is **refused**, because a search
can run for hours and its output is expensive to reproduce. Re-running the same
command stops with an error instead of silently discarding the previous run, and
names both ways forward:

```sh
t3v -o keys.csv foo
# error: keys.csv already exists; refusing to overwrite it.
#   --append  add to it, keeping what is already there
#   --overwrite  discard it and start over
```

`--append` adds to the file, which is what makes a long search resumable: run it
again with the same `--out` and the keys pile up. The result stays a single
well-formed file — a CSV keeps its one header rather than repeating it, and a
`.json` array has its closing bracket removed so the new records stay *inside*
the document. `--overwrite` replaces it, and only then is anything discarded.

Both require `--out`; per-key `--dst` output has no equivalent problem, since
each key is its own file and re-running simply rewrites the same ones.

Keys are written as they are found, so a search can be interrupted at any point
and everything already on disk is usable. At 5 characters and longer the tool
`fsync`s each key by default: those searches run for hours, a key arrives every
few seconds, and durability is free. Below 5 characters keys arrive thousands per
second and `fsync` is not. `--fsync` and `--no-fsync` override.

`--count N` stops after N keys and exits through the normal shutdown path, so the
output file is finished properly — a `.json` array still gets its closing bracket.
`--quiet` silences the progress line and the configuration summary, leaving only
the addresses.

## Benchmarking

One flag, a duration. Matching is on by default, and the two measurements
bracket the truth:

```sh
t3v --bench 2                # random 4-character prefix, 2 seconds
t3v --bench 5 abcd           # the prefix 'abcd', 5 seconds
t3v --bench 5 --no-match     # kernel only, no matching
```

**With matching** (the default) a real search runs and the rate is keys
**found** per second — the number that predicts how long a search takes. It is
also extrapolated to keys examined by multiplying by `2^(5·len)`, which is an
*expectation*, not a measurement: a search's yield is geometric, so a short run
can land above or below it. Give prefixes as positional arguments; with none
given, a random 4-character one is used per repetition, so an unlucky prefix
cannot flatter or spoil a run. All prefixes in one run must be the same length.

**With `--no-match`** the kernel is timed alone and the rate is keys
**examined** per second: the GPU's ceiling, with no matching, no batch
inversion and no host work. A prefix cannot be combined with it, since there is
nothing to match against.

`--bench` takes a duration and nothing else. `2` is a legal base32 prefix
character, so a flag accepting either would make `--bench 2` ambiguous between
"two seconds" and "the prefix 2"; prefixes are always positional instead.

On an RTX 3050 6GB laptop (20 SMs, 45 W cap) the match path sustains
**~223M keys/s**, confirmed three independent ways including a row count that
consults neither the launch counter nor the drop counter. See
[THROUGHPUT.md](THROUGHPUT.md) before quoting any rate from this repo.

Expected time to the first key at that rate:

| prefix | candidates | expected time |
|---|---|---|
| 3 | 2^15 | 0.0001 s |
| 4 | 2^20 | 0.005 s |
| 5 | 2^25 | 0.15 s |
| 6 | 2^30 | 4.8 s |
| 7 | 2^35 | 2.6 min |
| 8 | 2^40 | 1.4 h |
| 9 | 2^45 | 1.8 d |
| 10 | 2^50 | 58 d |
| 11 | 2^55 | 5.1 y |
| 12 | 2^60 | 164 y |

A 1- or 2-character search finds keys far faster than the host can derive and
write them, and the per-launch capture ring holds only 16 384 keys per prefix,
so most matches at those lengths are dropped. The progress line says so
explicitly rather than under-reporting.

## Tuning

Compile-time kernel constants live in [kernel.conf](kernel.conf), which carries
the measured sweep behind each value. Any of them can be overridden for a
one-off experiment, which wins over the file and needs no edit:

```sh
T3V_CHUNK=128 cargo +nightly build --release
```

| variable | default | purpose |
|---|---|---|
| `T3V_BLOCK` | 128 | upper bound on threads per block; the host derives the real value from register usage |
| `T3V_CHUNK` | 256 | candidates batch-inverted together; the inversion amortisation factor |
| `T3V_BATCH` | 512 | candidates per thread per launch; amortises SHA-512 and the base-point scalar multiplication |
| `T3V_COMB_W` | 4 | bits per base-point table window; only 2, 3 and 4 are legal |
| `T3V_PTX_PROFILE` | `release` | profile for the kernel crate |
| `T3V_PTX_RUSTFLAGS` | none | `RUSTFLAGS` for the kernel build only |
| `CUDA_PATH` | autodetected | CUDA root, used to find `libcudart` |

Invalid values are rejected on the main thread at startup, naming the variable,
rather than panicking inside a worker thread where the cause is invisible.

## How it works

```
src/         host: CLI, CUDA launch and readback, output sink, benchmarks
core/        device crate, compiled to PTX and embedded by build.rs
  field.rs     5x51 field arithmetic over u64 limbs
  group.rs     Edwards points, the comb table, batch inversion
  kernel.rs    the walk, the capture ring, `render`
  sha512.rs    the one hash a key needs
kernel.conf  compile-time tunables, with their measured sweeps
```

`build.rs` compiles `core/` for `nvptx64-nvidia-cuda` as a `cdylib`, which makes
rustc hand the LLVM bitcode to `llvm-bitcode-linker` and emit PTX, which the
binary embeds and loads at runtime. The tunables are passed to the device crate
through the nested build's environment *and* to the host crate through
`cargo:rustc-env`; setting only one of the two leaves the kernel and the host
that launches it disagreeing, which is the failure mode to watch for.

The kernel walks a chunk once, keeping each candidate's `Y` and `Z`, batch-inverts
the chunk's `Z` coordinates, and compresses from the result. A prefix is decided
by the low bits of `y`, so `X/Z` is never computed and neither is the rest of the
32-byte encoding. See [kernel.conf](kernel.conf) for the tuning constants and
[NOWALK.md](NOWALK.md) for why the walk happens once rather than twice -- note
that file's instruction counts are deprecated even though its measurements are
not.

## Disclaimer

Brand new and not thoroughly vetted. Please report bugs
[here](https://github.com/dr-bonez/tor-v3-vanity/issues).
