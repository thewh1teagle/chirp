# Chirp C

Native C/C++ Qwen3-TTS runtime and C API.

Build:

```console
cmake -S runtimes/qwen -B runtimes/qwen/build
cmake --build runtimes/qwen/build -j
```

Run:

```console
./runtimes/qwen/build/chirp-runtime \
  --model models/qwen3-tts-0.6b-f16.gguf \
  --codec models/qwen3-tts-tokenizer-f16.gguf \
  --text "hello" \
  --ref tmp/refs/female1.wav \
  --output tmp/chirp_runtime.wav
```

Text tokenization is native and read from the embedded tokenizer metadata in the
model GGUF. Reference speaker embedding is also native, so inference does not
invoke Python.

The runtime fetches `ggml`, `dr_wav`, `kissfft`, and `libsoxr` with CMake
`FetchContent`. `libsoxr` handles reference WAV resampling and `kissfft` handles
the speaker encoder STFT.
