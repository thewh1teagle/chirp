# Chirp Agent Skill

Use Chirp when you need local, offline text-to-speech with optional voice cloning. Chirp provides a small native runner that can create WAV files from text using Qwen3-TTS.

## Requirements

- A platform runner from the `chirp-runner` release.
- The model bundle from the `chirp-models` release.
- Local filesystem access to the model GGUF, codec GGUF, output path, and optional reference WAV.

Recommended versions:

```console
chirp-runner-v0.2.0
chirp-models-v0.1.3
```

## Install

Download the runner for the current platform:

```console
wget https://github.com/thewh1teagle/chirp/releases/download/chirp-runner-v0.2.0/chirp-runner-darwin-arm64.tar.gz
tar -xzf chirp-runner-darwin-arm64.tar.gz
```

Download the models once:

```console
wget https://github.com/thewh1teagle/chirp/releases/download/chirp-models-v0.1.3/chirp-models-q5_0.tar.gz
mkdir -p models
tar -xzf chirp-models-q5_0.tar.gz -C models
```

Expected model files:

```console
models/qwen3-tts-0.6b-q5_0.gguf
models/qwen3-tts-tokenizer-q5_0.gguf
```

If filenames differ, inspect the extracted model directory and pass the actual GGUF paths.

## List Languages

Before creating speech, list supported language names:

```console
./chirp-runner-darwin-arm64/chirp languages \
  --model models/qwen3-tts-0.6b-q5_0.gguf \
  --codec models/qwen3-tts-tokenizer-q5_0.gguf
```

Supported languages are currently:

```console
auto
chinese
english
french
german
italian
japanese
korean
portuguese
russian
spanish
```

Use `auto` when unsure.

## Create WAV With CLI

Use the CLI for one-shot generation:

```console
./chirp-runner-darwin-arm64/chirp speak \
  --model models/qwen3-tts-0.6b-q5_0.gguf \
  --codec models/qwen3-tts-tokenizer-q5_0.gguf \
  --text "Hello from Chirp." \
  --language english \
  --output output.wav
```

For voice cloning, pass a reference WAV:

```console
./chirp-runner-darwin-arm64/chirp speak \
  --model models/qwen3-tts-0.6b-q5_0.gguf \
  --codec models/qwen3-tts-tokenizer-q5_0.gguf \
  --text "Hello with a reference voice." \
  --language english \
  --ref reference.wav \
  --output output.wav
```

The output is a WAV file.

## Preset Voices

Bundled voice presets are listed in the Chirp source tree at:

```console
chirp-desktop/src/assets/voices.json
```

Each entry includes an `id`, `name`, `description`, `lang`, and `url`. Download the preset WAV from `url`, then pass the downloaded local file path as `voice_reference`.

## Use HTTP Server

Use the server for repeated requests because it keeps the model loaded:

```console
./chirp-runner-darwin-arm64/chirp serve \
  --host 127.0.0.1 \
  --port 8080 \
  --model models/qwen3-tts-0.6b-q5_0.gguf \
  --codec models/qwen3-tts-tokenizer-q5_0.gguf
```

Create speech:

```console
curl -sS http://127.0.0.1:8080/v1/audio/speech \
  -H 'content-type: application/json' \
  -d '{"input":"Hello from the HTTP API.","language":"english","response_format":"wav"}' \
  --output output.wav
```

Create speech with a reference WAV:

```console
curl -sS http://127.0.0.1:8080/v1/audio/speech \
  -H 'content-type: application/json' \
  -d '{"input":"Hello with a reference voice.","language":"english","voice_reference":"reference.wav","response_format":"wav"}' \
  --output output.wav
```

Useful endpoints:

```console
GET /health
GET /v1/models
GET /v1/languages
POST /v1/audio/speech
POST /v1/models/load
DELETE /v1/models
```

## Generation Controls

The runner accepts these optional generation controls when loading the model:

```console
--max-tokens 0
--temperature 0.9
--top-k 50
```

For most agent workflows, keep defaults. `--max-tokens` is a maximum generated frame budget, not exact progress.

## Troubleshooting

- If the runner says no model is loaded, pass both `--model` and `--codec`, or call `POST /v1/models/load`.
- If language is rejected, call `languages` or `GET /v1/languages` and use one of the returned names.
- If voice cloning fails, ensure the reference is a readable local WAV path.
- If first generation is slow, that is usually model and GPU backend initialization.
- If the process prints GPU/backend logs to stderr, ignore them unless the command exits non-zero.
