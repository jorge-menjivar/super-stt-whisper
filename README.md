# Super STT — Whisper backend

[![coverage](https://img.shields.io/endpoint?url=https://jorge-menjivar.github.io/super-stt-whisper/coverage.json)](https://jorge-menjivar.github.io/super-stt-whisper/)

A speech-to-text backend for **[Super STT](https://github.com/jorge-menjivar/super-stt)**.
It runs [OpenAI's Whisper](https://huggingface.co/openai) models locally — on CPU
or a CUDA GPU — to turn speech into text.

Super STT is an on-device speech-to-text engine. It doesn't ship any models of
its own — it loads **backends** like this one at runtime. This repo packages the
Whisper family (tiny through large) as one of those backends.

## Using it

You don't run this directly. Super STT discovers it through its backend
registry, downloads a prebuilt release for your platform, fetches the model
weights, and runs it sandboxed. To use Whisper, install Super STT and enable it
from the app — see the [Super STT docs](https://github.com/jorge-menjivar/super-stt).

## Models

Chosen by `name` when Super STT loads the backend. Each runs on CPU or a CUDA
GPU; weights are pulled from Hugging Face on first load. The `.en` variants are
English-only; the rest are multilingual. `~VRAM` is the GPU memory the model is
expected to use.

| Model (`name`)       | Upstream model                                                       | Device     | ~VRAM  |
| -------------------- | ------------------------------------------------------------------- | ---------- | ------ |
| `whisper-tiny`       | [openai/whisper-tiny](https://huggingface.co/openai/whisper-tiny)             | CPU / CUDA | ~1 GB  |
| `whisper-tiny.en`    | [openai/whisper-tiny.en](https://huggingface.co/openai/whisper-tiny.en)       | CPU / CUDA | ~1 GB  |
| `whisper-base`       | [openai/whisper-base](https://huggingface.co/openai/whisper-base)             | CPU / CUDA | ~1 GB  |
| `whisper-base.en`    | [openai/whisper-base.en](https://huggingface.co/openai/whisper-base.en)       | CPU / CUDA | ~1 GB  |
| `whisper-small`      | [openai/whisper-small](https://huggingface.co/openai/whisper-small)           | CPU / CUDA | ~2 GB  |
| `whisper-small.en`   | [openai/whisper-small.en](https://huggingface.co/openai/whisper-small.en)     | CPU / CUDA | ~2 GB  |
| `whisper-medium`     | [openai/whisper-medium](https://huggingface.co/openai/whisper-medium)         | CPU / CUDA | ~5 GB  |
| `whisper-medium.en`  | [openai/whisper-medium.en](https://huggingface.co/openai/whisper-medium.en)   | CPU / CUDA | ~5 GB  |
| `whisper-large`      | [openai/whisper-large](https://huggingface.co/openai/whisper-large)           | CPU / CUDA | ~10 GB |

## What's in here

A small, self-contained Rust program that loads a Whisper model and speaks the
Super STT backend protocol (a tiny HTTP API over a Unix socket). It shares no
code with the Super STT project.

## Building from source

Most people never need to — Super STT downloads prebuilt releases. For
development (requires [`just`](https://github.com/casey/just)):

```bash
just build-release                  # CPU build
just build-release --features cuda  # GPU build (needs a CUDA toolkit)
just ci                             # format, lint, build, and test
```

## License

GPL-3.0-only.
