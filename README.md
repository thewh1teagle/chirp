<p align="center">
  <a target="_blank" href="https://thewh1teagle.github.io/chirp/">
    <img width="96" alt="Chirp logo" src="./chirp-desktop/src/assets/chirp-logo.svg" />
  </a>
</p>

<h1 align="center">Chirp</h1>

<p align="center">
  <strong>Native offline Qwen3-TTS for desktop</strong>
  <br />
</p>

<p align="center">
  <a target="_blank" href="https://thewh1teagle.github.io/chirp/">
    🔗 Download Chirp
  </a>
  &nbsp; | &nbsp; Give it a Star ⭐ | &nbsp;
  <a target="_blank" href="https://github.com/sponsors/thewh1teagle">Support the project 🤝</a>
</p>

<hr />

<p align="center">
  <a target="_blank" href="https://thewh1teagle.github.io/chirp/">
    <img width="800" alt="Chirp desktop screenshot" src="https://github.com/user-attachments/assets/6f8fca35-bf3b-4bdb-bdcc-1c52f535e04d" />
  </a>
</p>

## Features

- Local text-to-speech with Qwen3-TTS
- Fully offline generation after the model is downloaded
- Voice cloning from a reference WAV
- Create speech in 10 supported languages
- Audio preview after creation
- 💻 Desktop support for `macOS`, `Windows`, and `Linux`
- 🎮 Optimized for `Nvidia` / `AMD` / `Intel` GPUs
- 🍎 Optimized desktop builds for Apple Silicon macOS
- CLI support for creating WAV files
- Local HTTP API with Swagger docs for tools and automation
- Agent-ready `/skill` instructions and voice preset catalog for AI workflows

## Layout

```console
chirp-c/   C/C++ runtime, C API, GGUF conversion scripts
chirp-runner/ Go CLI and HTTP server using chirp-c through cgo
chirp-desktop/ Tauri + React desktop app
plans/     implementation notes and validation logs
```

## Build

See [BUILDING.md](docs/BUILDING.md).
