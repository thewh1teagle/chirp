#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "voices" / "kokoro"
BASE_URL = "https://hexgrad-kokoro-tts.hf.space"
TEXT = "Failure doesn't mean you are a failure it just means you haven't succeeded yet."


def request_json(url: str, data: dict[str, Any] | None = None) -> Any:
    body = None
    headers = {"User-Agent": "chirp-voice-generator/1.0"}
    if data is not None:
        body = json.dumps(data).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=body, headers=headers)
    with urllib.request.urlopen(req, timeout=120) as res:
        return json.loads(res.read().decode("utf-8"))


def download(url: str, dest: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "chirp-voice-generator/1.0"})
    with urllib.request.urlopen(req, timeout=120) as res:
        dest.write_bytes(res.read())


def voices_from_config(config: dict[str, Any]) -> list[dict[str, str]]:
    for component in config["components"]:
        if component.get("id") != 24:
            continue
        voices = []
        for label, value in component["props"]["choices"]:
            clean = re.sub(r"[^\w\s-]", "", label).strip()
            clean = re.sub(r"\s+", " ", clean)
            voices.append(
                {
                    "id": value,
                    "name": clean,
                    "description": f"Kokoro reference voice {clean}.",
                    "language": "English",
                    "url": "",
                }
            )
        return voices
    raise RuntimeError("voice dropdown component not found in config")


def generate_voice(voice_id: str, fn_index: int) -> str:
    session_hash = str(uuid.uuid4())
    request_json(
        f"{BASE_URL}/gradio_api/queue/join",
        {
            "data": [TEXT, voice_id, 1, True],
            "fn_index": fn_index,
            "session_hash": session_hash,
        },
    )

    url = f"{BASE_URL}/gradio_api/queue/data?{urllib.parse.urlencode({'session_hash': session_hash})}"
    req = urllib.request.Request(url, headers={"User-Agent": "chirp-voice-generator/1.0"})
    with urllib.request.urlopen(req, timeout=300) as res:
        for raw in res:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data: "):
                continue
            event = json.loads(line.removeprefix("data: "))
            msg = event.get("msg")
            if msg == "process_completed":
                if not event.get("success", False):
                    raise RuntimeError(f"generation failed for {voice_id}: {event}")
                return extract_audio_url(event)
            if msg == "queue_full":
                raise RuntimeError(f"queue full for {voice_id}")
    raise RuntimeError(f"no completion event for {voice_id}")


def extract_audio_url(event: dict[str, Any]) -> str:
    output = event.get("output", {})
    data = output.get("data", [])
    if not data:
        raise RuntimeError(f"missing output data: {event}")
    audio = data[0]
    if not isinstance(audio, dict):
        raise RuntimeError(f"unexpected audio output: {audio}")
    url = audio.get("url")
    if not url:
        path = audio.get("path")
        if not path:
            raise RuntimeError(f"missing audio url/path: {audio}")
        url = f"{BASE_URL}/gradio_api/file={urllib.parse.quote(path)}"
    return str(url)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    config = request_json(f"{BASE_URL}/config")
    voices = voices_from_config(config)
    fn_index = 4

    for index, voice in enumerate(voices, start=1):
        dest = OUT_DIR / f"{voice['id']}.wav"
        if dest.exists() and dest.stat().st_size > 0:
            print(f"[{index}/{len(voices)}] exists {dest.name}")
            continue
        print(f"[{index}/{len(voices)}] generating {voice['id']}...")
        for attempt in range(1, 4):
            try:
                audio_url = generate_voice(voice["id"], fn_index)
                download(audio_url, dest)
                print(f"  wrote {dest}")
                break
            except (urllib.error.URLError, TimeoutError, RuntimeError) as exc:
                if attempt == 3:
                    raise
                print(f"  retry {attempt}: {exc}")
                time.sleep(4 * attempt)

    catalog = {
        "version": "chirp-voices-v0.1.0",
        "source": "hexgrad/Kokoro-TTS",
        "text": TEXT,
        "voices": [
            {
                **voice,
                "url": f"https://github.com/thewh1teagle/chirp/releases/download/chirp-voices-v0.1.0/{voice['id']}.wav",
            }
            for voice in voices
        ],
    }
    (OUT_DIR / "voices.json").write_text(json.dumps(catalog, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT_DIR / 'voices.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
