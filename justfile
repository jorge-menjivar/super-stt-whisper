# SPDX-License-Identifier: GPL-3.0-only
# Task runner for the standalone Whisper backend. Mirrors the recipe names used
# by the main super-stt repo (`just check`, etc.).

# Default: build release
default: build-release

# Compiles with debug profile. Usage: just build-debug [--features cuda]
build-debug *args:
    cargo build {{ args }}

# Compiles with release profile. Usage: just build-release [--features cuda]
build-release *args:
    cargo build --release --locked {{ args }}

# Runs a clippy check — mirrors super-stt's lint. There, `--all-features
# --workspace` enables no CUDA (workspace crates have no cuda feature; the GPU
# backends are out-of-tree), so the equivalent here is a default-feature (CPU)
# lint, which still covers all of whisper's own code. Run `just check
# --all-features` locally to additionally lint the candle CUDA backend (needs a
# CUDA toolkit).
check *args:
    cargo clippy --all-targets {{ args }} -- -W clippy::pedantic -D warnings -D unused_must_use

# Runs a clippy check with JSON message format (consumed by clippy-sarif in CI)
check-json: (check '--message-format=json')

# Apply rustfmt to the whole crate
fmt:
    cargo fmt --all

# Check formatting without modifying files
fmt-check:
    cargo fmt --all -- --check

# Run the test suite. Usage: just test [--verbose]
test *args:
    cargo test --locked {{ args }}

# Run doctests. Unlike the bin-only backends, this crate has a lib target, so
# `cargo test --doc` has something to run.
doctest:
    cargo test --locked --doc

# Measure code coverage (requires cargo-llvm-cov). --remap-path-prefix keeps the
# report paths relative (src/...), and tests/ is excluded so only product code
# is counted. Usage: just coverage [--html]
coverage *args:
    cargo llvm-cov --locked --remap-path-prefix --ignore-filename-regex 'tests/' {{ args }}

# Coverage for CI: write lcov.info and print a summary.
coverage-lcov:
    cargo llvm-cov --locked --remap-path-prefix --ignore-filename-regex 'tests/' --lcov --output-path lcov.info
    cargo llvm-cov report --summary-only --ignore-filename-regex 'tests/'

# Full local CI gate: format, lint, build, test, doctest
ci: fmt-check check build-release test doctest
