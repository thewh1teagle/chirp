package server

import (
	"context"
	"fmt"
	"net/http"

	"github.com/danielgtaylor/huma/v2"
	"github.com/danielgtaylor/huma/v2/adapters/humago"
)

type docsHealthOutput struct {
	Body struct {
		Status string `json:"status" example:"ready"`
		Loaded bool   `json:"loaded" example:"true"`
		Model  string `json:"model,omitempty" example:"qwen3-tts-q5_0.gguf"`
	}
}

type docsModelsOutput struct {
	Body struct {
		Loaded bool   `json:"loaded" example:"true"`
		Model  string `json:"model,omitempty" example:"qwen3-tts-q5_0.gguf"`
		Path   string `json:"path,omitempty" example:"/path/to/qwen3-tts-q5_0.gguf"`
		Codec  string `json:"codec,omitempty" example:"/path/to/qwen3-tts-codec-q5_0.gguf"`
	}
}

type docsLanguagesOutput struct {
	Body struct {
		Languages []string         `json:"languages" example:"auto,english,spanish"`
		Items     []map[string]any `json:"items"`
	}
}

type docsModelLoadInput struct {
	Body struct {
		ModelPath   string  `json:"model_path,omitempty" example:"/path/to/qwen3-tts-q5_0.gguf"`
		CodecPath   string  `json:"codec_path,omitempty" example:"/path/to/qwen3-tts-codec-q5_0.gguf"`
		MaxTokens   int     `json:"max_tokens,omitempty" example:"0"`
		Temperature float32 `json:"temperature,omitempty" example:"0.9"`
		TopK        int     `json:"top_k,omitempty" example:"50"`
	}
}

type docsModelLoadOutput struct {
	Body struct {
		Status string `json:"status" example:"loaded"`
		Model  string `json:"model" example:"qwen3-tts-q5_0.gguf"`
	}
}

type docsStatusOutput struct {
	Body struct {
		Status string `json:"status" example:"unloaded"`
	}
}

type docsSkillOutput struct {
	Body string `contentType:"text/markdown"`
}

type docsSpeechInput struct {
	Body struct {
		Input          string `json:"input" example:"Hello from Chirp."`
		VoiceReference string `json:"voice_reference,omitempty" example:"/path/to/reference.wav"`
		ResponseFormat string `json:"response_format,omitempty" example:"wav"`
		Language       string `json:"language,omitempty" example:"auto"`
	}
}

type docsSpeechOutput struct {
	Body []byte `contentType:"audio/wav"`
}

func (s *Server) registerDocsRoutes(mux *http.ServeMux) {
	docsMux := http.NewServeMux()
	config := huma.DefaultConfig("Chirp API", s.Version)
	config.Info.Description = "Local Qwen3-TTS HTTP API served by the Chirp runner."
	config.DocsPath = ""
	api := humago.New(docsMux, config)

	huma.Register(api, huma.Operation{
		Method:      http.MethodGet,
		Path:        "/health",
		OperationID: "healthCheck",
		Summary:     "Check runner health",
		Description: "Returns whether the runner is alive and whether a model is loaded.",
	}, func(context.Context, *struct{}) (*docsHealthOutput, error) {
		return nil, huma.Error501NotImplemented("spec-only operation")
	})

	huma.Register(api, huma.Operation{
		Method:      http.MethodGet,
		Path:        "/skill",
		OperationID: "getAgentSkill",
		Summary:     "Get agent skill prompt",
		Description: "Returns Markdown instructions for AI agents that want to use this local Chirp API.",
	}, func(context.Context, *struct{}) (*docsSkillOutput, error) {
		return nil, huma.Error501NotImplemented("spec-only operation")
	})

	huma.Register(api, huma.Operation{
		Method:      http.MethodGet,
		Path:        "/v1/models",
		OperationID: "getModelState",
		Summary:     "Get model state",
	}, func(context.Context, *struct{}) (*docsModelsOutput, error) {
		return nil, huma.Error501NotImplemented("spec-only operation")
	})

	huma.Register(api, huma.Operation{
		Method:      http.MethodPost,
		Path:        "/v1/models/load",
		OperationID: "loadModel",
		Summary:     "Load a Qwen3-TTS model",
		Description: "Loads explicit model_path and codec_path when provided. If both are omitted, loads the default model paths from CHIRP_MODEL_PATH and CHIRP_CODEC_PATH.",
	}, func(context.Context, *docsModelLoadInput) (*docsModelLoadOutput, error) {
		return nil, huma.Error501NotImplemented("spec-only operation")
	})

	huma.Register(api, huma.Operation{
		Method:      http.MethodDelete,
		Path:        "/v1/models",
		OperationID: "unloadModel",
		Summary:     "Unload the current model",
	}, func(context.Context, *struct{}) (*docsStatusOutput, error) {
		return nil, huma.Error501NotImplemented("spec-only operation")
	})

	huma.Register(api, huma.Operation{
		Method:      http.MethodGet,
		Path:        "/v1/languages",
		OperationID: "listLanguages",
		Summary:     "List supported synthesis languages",
	}, func(context.Context, *struct{}) (*docsLanguagesOutput, error) {
		return nil, huma.Error501NotImplemented("spec-only operation")
	})

	huma.Register(api, huma.Operation{
		Method:      http.MethodPost,
		Path:        "/v1/audio/speech",
		OperationID: "createSpeech",
		Summary:     "Create speech",
		Description: "Synthesizes input text to WAV audio using the loaded model.",
	}, func(context.Context, *docsSpeechInput) (*docsSpeechOutput, error) {
		return nil, huma.Error501NotImplemented("spec-only operation")
	})

	mux.HandleFunc("GET /docs", serveSwaggerUI)
	mux.HandleFunc("GET /docs/", redirectDocs)
	mux.Handle("/openapi.json", docsMux)
	mux.Handle("/openapi.yaml", docsMux)
	mux.Handle("/openapi-3.0.json", docsMux)
	mux.Handle("/openapi-3.0.yaml", docsMux)
	mux.Handle("/schemas/", docsMux)
}

func redirectDocs(w http.ResponseWriter, r *http.Request) {
	http.Redirect(w, r, "/docs", http.StatusTemporaryRedirect)
}

func serveSwaggerUI(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	fmt.Fprint(w, `<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Chirp API Docs</title>
    <link rel="stylesheet" href="https://unpkg.com/swagger-ui-dist@5/swagger-ui.css" />
    <style>
      html, body { margin: 0; padding: 0; background: #f7f8fb; }
      #swagger-ui { max-width: 1200px; margin: 0 auto; }
    </style>
  </head>
  <body>
    <div id="swagger-ui"></div>
    <script src="https://unpkg.com/swagger-ui-dist@5/swagger-ui-bundle.js" crossorigin></script>
    <script>
      SwaggerUIBundle({
        url: "/openapi.json",
        dom_id: "#swagger-ui",
        deepLinking: true,
        displayRequestDuration: true
      });
    </script>
  </body>
</html>`)
}
