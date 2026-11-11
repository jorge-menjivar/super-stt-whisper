# Super Voice — Whisper backend

[![coverage](https://img.shields.io/endpoint?url=https://super-libre.github.io/super-voice-whisper/coverage.json)](https://super-libre.github.io/super-voice-whisper/)

OpenAI's [Whisper](https://huggingface.co/openai) speech recognition models,
from `whisper-tiny` to `whisper-large` and their English-only `.en` variants,
as a subprocess backend for
[Super Voice](https://github.com/jorge-menjivar/super-voice). On Linux, on the
CPU or a CUDA GPU. It hears your dictation context: the vocabulary and prompt
you set in Super Voice are the text Whisper hears as coming before your audio,
so it spells your terms the way you do.

## What it is

The Super Voice daemon does not compile model inference in-tree. It discovers
backends on disk and drives each one over a `/v1` HTTP contract on a Unix
socket. This is one such backend: a single Rust binary that runs the models
through [candle](https://github.com/huggingface/candle) and answers
`POST /v1/transcribe`.

## License

GPL-3.0-only. See `LICENSE`.
