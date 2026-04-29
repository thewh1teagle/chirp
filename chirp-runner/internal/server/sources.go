package server

import (
	"encoding/json"
	"net/http"
)

const (
	qwenModelsTag    = "chirp-models-v0.1.3"
	qwenModelFile    = "qwen3-tts-model.gguf"
	qwenCodecFile    = "qwen3-tts-codec.gguf"
	qwenModelBaseURL = "https://huggingface.co/thewh1teagle/qwen3-tts-gguf/resolve/main"
	kokoroModelsTag  = "kokoro-v1.0"
	kokoroModelDir   = "chirp-kokoro-models-kokoro-v1.0"
	kokoroBundleURL  = "https://huggingface.co/thewh1teagle/chirp-kokoro-models/resolve/main/chirp-kokoro-models-kokoro-v1.0.tar.gz"
	voicesCatalogURL = "https://raw.githubusercontent.com/thewh1teagle/chirp/main/chirp-desktop/src/assets/voices.json"
)

type modelSourceFile struct {
	Name string `json:"name"`
	URL  string `json:"url"`
}

type modelSource struct {
	ID          string            `json:"id"`
	Name        string            `json:"name"`
	Version     string            `json:"version"`
	Recommended bool              `json:"recommended,omitempty"`
	Size        string            `json:"size"`
	Description string            `json:"description"`
	Files       []modelSourceFile `json:"files,omitempty"`
	ArchiveURL  string            `json:"archive_url,omitempty"`
	Directory   string            `json:"directory,omitempty"`
}

type modelSourcesResponse struct {
	Runtimes     []modelSource `json:"runtimes"`
	VoicesURL    string        `json:"voices_url"`
	DefaultPaths []string      `json:"default_paths"`
}

func modelSources() modelSourcesResponse {
	return modelSourcesResponse{
		Runtimes: []modelSource{
			{
				ID:          "qwen",
				Name:        "Qwen",
				Version:     qwenModelsTag,
				Size:        "~900 MB",
				Description: "Voice cloning, multilingual synthesis, best quality on supported GPU hardware.",
				Files: []modelSourceFile{
					{Name: qwenModelFile, URL: qwenModelBaseURL + "/" + qwenModelFile},
					{Name: qwenCodecFile, URL: qwenModelBaseURL + "/" + qwenCodecFile},
				},
				Directory: "chirp-models-q5_0",
			},
			{
				ID:          "kokoro",
				Name:        "Kokoro",
				Version:     kokoroModelsTag,
				Recommended: true,
				Size:        "~336 MB",
				Description: "Fast local multi-voice speech with a lighter model bundle.",
				ArchiveURL:  kokoroBundleURL,
				Directory:   kokoroModelDir,
			},
		},
		VoicesURL: voicesCatalogURL,
		DefaultPaths: []string{
			"macOS: ~/Library/Application Support/com.thewh1teagle.chirp/models",
			"Windows: %LOCALAPPDATA%\\com.thewh1teagle.chirp\\models",
			"Linux: ~/.local/share/com.thewh1teagle.chirp/models",
		},
	}
}

func (s *Server) handleModelSources(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(modelSources())
}
