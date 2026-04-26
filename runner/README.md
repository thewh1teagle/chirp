# Chirp Runner

Small Go CLI and HTTP runner for the Qwen3-TTS C++ runtime.

Build the native C runtime first:

```bash
cmake -S chirp/chirp-c -B chirp/chirp-c/build
cmake --build chirp/chirp-c/build -j
```

Run one synthesis:

```bash
cd chirp/runner
go run ./cmd/chirp speak \
  --model ../../models/qwen3-tts-0.6b-q5_0.gguf \
  --codec ../../models/qwen3-tts-tokenizer-f16.gguf \
  --text "hello" \
  --ref ../../tmp/refs/female1.wav \
  --output ../../tmp/chirp_runner.wav
```

Start the HTTP server:

```bash
cd chirp/runner
go run ./cmd/chirp serve \
  --model ../../models/qwen3-tts-0.6b-q5_0.gguf \
  --codec ../../models/qwen3-tts-tokenizer-f16.gguf
```

Generate speech:

```bash
curl -sS http://127.0.0.1:PORT/v1/audio/speech \
  -H 'content-type: application/json' \
  -d '{"input":"hello","voice_reference":"../../tmp/refs/female1.wav"}' \
  --output speech.wav
```
