package server

import (
	"fmt"
	"net/http"
	"strings"
)

const skillTemplate = `# Chirp Local TTS API

You are using Chirp, a local Qwen3-TTS HTTP API.

Base URL: {{base_url}}
OpenAPI schema: {{base_url}}/openapi.json
Swagger docs: {{base_url}}/docs
Model sources: {{base_url}}/v1/models/sources

Before calling the API, fetch the OpenAPI schema from /openapi.json and use it as the source of truth for request and response shapes.

Recommended flow:

1. Call GET /health.
2. If loaded=false, call GET /v1/models/sources to discover runtimes, model download URLs, and default Chirp Desktop model locations.
3. Check whether the model files already exist in Chirp Desktop's default model directory:
   - macOS: ~/Library/Application Support/com.thewh1teagle.chirp/models
   - Windows: %LOCALAPPDATA%\com.thewh1teagle.chirp\models
   - Linux: ~/.local/share/com.thewh1teagle.chirp/models
4. Call POST /v1/models/load with an empty JSON object. Chirp Desktop passes the selected default model paths to the runner.
5. Call POST /v1/audio/speech to synthesize speech.
6. Send JSON with input, optional voice_reference, language, and response_format set to wav.
7. Save the audio/wav response to a .wav file.

Bundled voice presets are listed at https://raw.githubusercontent.com/thewh1teagle/chirp/main/chirp-desktop/src/assets/voices.json.
Use the url for a preset voice to download a reference WAV, then pass the downloaded local path as voice_reference.

Example:

~~~sh
curl {{base_url}}/health

curl {{base_url}}/v1/models/sources

curl -X POST {{base_url}}/v1/models/load \
  -H 'Content-Type: application/json' \
  -d '{}'

curl -X POST {{base_url}}/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -o speech.wav \
  -d '{"input":"Hello from Chirp","language":"auto","response_format":"wav"}'
~~~

If the API returns no_model, ask the user to install the Chirp model in the desktop app first.
`

func (s *Server) handleSkill(w http.ResponseWriter, r *http.Request) {
	baseURL := "http://" + r.Host
	w.Header().Set("Content-Type", "text/markdown; charset=utf-8")
	fmt.Fprint(w, strings.ReplaceAll(skillTemplate, "{{base_url}}", baseURL))
}
