# Qwen3-TTS GGUF tools

Python tools for converting the upstream Hugging Face Qwen3-TTS checkpoints into
the GGUF files consumed by `qwentts-rs`.

These tools are offline model preparation utilities. They are not part of the
Rust crate build.

## Setup

Run commands from this directory:

```console
cd crates/qwentts-rs/tools/gguf
uv sync
```

The converters depend on `gguf`, `torch`, `safetensors`, `transformers`, `numpy`,
and `tqdm`. The dependency set is locked in `uv.lock`.

## Download upstream checkpoint

```console
huggingface-cli download Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  --local-dir models/Qwen3-TTS-12Hz-0.6B-Base
```

The tokenizer checkpoint is expected under:

```text
models/Qwen3-TTS-12Hz-0.6B-Base/speech_tokenizer
```

## Convert model

```console
uv run python convert_model_to_gguf.py \
  --input models/Qwen3-TTS-12Hz-0.6B-Base \
  --output models/qwen3-tts-0.6b-f16.gguf \
  --type f16
```

Supported output types:

```text
f16, f32, q8_0, q5_0, q6_k, q4_k
```

By default, quality-sensitive tensors such as embeddings and heads are kept in
F16 when quantizing. Use `--quantize-all` only when intentionally quantizing
every rank > 1 tensor.

## Convert codec/tokenizer

```console
uv run python convert_codec_to_gguf.py \
  --input models/Qwen3-TTS-12Hz-0.6B-Base/speech_tokenizer \
  --output models/qwen3-tts-tokenizer-f16.gguf \
  --type f16
```

Supported codec output types:

```text
f16, f32, q8_0, q5_0
```

## Notes

- The converters expect Hugging Face safetensors checkpoints and config files.
- The generated GGUF metadata uses the tensor names expected by `qwentts-rs`.
- Keep generated model files out of git unless they are intentionally published
  as release assets.
