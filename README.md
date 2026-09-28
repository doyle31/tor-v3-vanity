# tor-v3-vanity
A TOR v3 vanity url generator designed to run on an NVIDIA GPU.

Disclaimer: This project is brand new and hasn't been thoroughly vetted.
Please report any bugs you find [here](https://github.com/dr-bonez/tor-v3-vanity/issues).

The program is designed to use all available cuda devices, and will automatically decide the number of threads and blocks to use.

Now supports multiple prefixes!

## Installation

- [Install Rust](https://rustup.rs)
- [Install Cuda](https://developer.nvidia.com/cuda-downloads)
- `rustup toolchain install nightly`
- `rustup component add llvm-bitcode-linker --toolchain nightly`
- `rustup target add nvptx64-nvidia-cuda --toolchain nightly`
- `git clone https://github.com/dr-bonez/tor-v3-vanity`
- `cd tor-v3-vanity`
- `cargo +nightly install --path .`

The kernel crate (`core/`) is compiled for `nvptx64-nvidia-cuda` with
`--crate-type cdylib`, which makes rustc link the LLVM bitcode with
`llvm-bitcode-linker` and emit the PTX that the binary embeds. This all
happens in `build.rs`; `llvm-bitcode-linker` and the `nvptx64-nvidia-cuda`
target must be installed for the nightly toolchain listed above.

If CUDA is installed somewhere other than `/usr/local/cuda` or `/opt/cuda`,
point the build at it with `CUDA_PATH`:

- `CUDA_PATH=/path/to/cuda cargo +nightly install --path .`

### Build options

| Variable             | Default            | Purpose                                              |
| -------------------- | ------------------ | ---------------------------------------------------- |
| `CUDA_PATH`          | autodetected       | CUDA install root, used to find `libcudart`          |
| `T3V_PTX_PROFILE`    | `release`          | Profile used to build the kernel crate               |
| `T3V_PTX_RUSTFLAGS`  | none               | `RUSTFLAGS` for the kernel build only                |

## Usage

- Create output dir
  - `mkdir mykeys`
- Run `t3v`
  - `t3v --dst mykeys/ myprefix1,myprefix2`
- Use the resulting file as your `hs_ed25519_secret_key`
  - `cat mykeys/myprefixwhatever.onion > /var/lib/tor/hidden_service/hs_ed25519_secret_key`

## Bench
On my 1070ti, I get the following time estimates:

| Prefix Length | Time       |
| ------------- | ---------- |
|             5 | 7 minutes  |
|             6 | 3.5 hours  |
|             7 | 5 days     |
|             8 | 22.5 weeks |
|             9 | 14 years   |
