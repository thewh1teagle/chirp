package server

import (
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"os"

	"github.com/thewh1teagle/chirp/chirp-runner/internal/chirpc"
	"github.com/thewh1teagle/chirp/chirp-runner/internal/kokoroc"
)

func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	s.mu.Lock()
	loaded := s.ctx != nil
	name := s.modelName
	runtimeName := s.runtime
	s.mu.Unlock()
	status := "ok"
	if loaded {
		status = "ready"
	}
	_ = json.NewEncoder(w).Encode(map[string]any{
		"status":  status,
		"loaded":  loaded,
		"model":   name,
		"runtime": runtimeName,
	})
}

func (s *Server) handleModels(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	s.mu.Lock()
	defer s.mu.Unlock()
	_ = json.NewEncoder(w).Encode(map[string]any{
		"loaded":  s.ctx != nil,
		"runtime": s.runtime,
		"model":   s.modelName,
		"path":    s.modelPath,
		"codec":   s.codecPath,
	})
}

func (s *Server) handleLanguages(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	languages := s.Languages()
	if languages == nil {
		writeError(w, http.StatusServiceUnavailable, errNoModel, "no model loaded")
		return
	}
	names := make([]string, 0, len(languages)+1)
	names = append(names, "auto")
	for _, language := range languages {
		names = append(names, language.Name)
	}
	_ = json.NewEncoder(w).Encode(map[string]any{
		"languages": names,
		"items":     languages,
	})
}

func (s *Server) handleModelLoad(w http.ResponseWriter, r *http.Request) {
	var body struct {
		Runtime   string `json:"runtime,omitempty"`
		ModelPath string `json:"model_path"`
		CodecPath string `json:"codec_path"`
		Qwen      struct {
			ModelPath   string  `json:"model_path"`
			CodecPath   string  `json:"codec_path"`
			MaxTokens   int     `json:"max_tokens,omitempty"`
			Temperature float32 `json:"temperature,omitempty"`
			TopK        int     `json:"top_k,omitempty"`
		} `json:"qwen,omitempty"`
		Kokoro struct {
			ModelPath      string  `json:"model_path"`
			VoicesPath     string  `json:"voices_path"`
			EspeakDataPath string  `json:"espeak_data_path"`
			Voice          string  `json:"voice,omitempty"`
			Language       string  `json:"language,omitempty"`
			Speed          float32 `json:"speed,omitempty"`
		} `json:"kokoro,omitempty"`
		MaxTokens   int     `json:"max_tokens,omitempty"`
		Temperature float32 `json:"temperature,omitempty"`
		TopK        int     `json:"top_k,omitempty"`
	}
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil && !errors.Is(err, io.EOF) {
		writeError(w, http.StatusBadRequest, errInvalidRequest, "invalid JSON request body")
		return
	}
	if body.Runtime == "" {
		body.Runtime = os.Getenv("CHIRP_RUNTIME")
	}
	if body.Runtime == "" && body.Kokoro.ModelPath != "" {
		body.Runtime = "kokoro"
	}
	if body.Runtime == "" && body.Qwen.CodecPath != "" {
		body.Runtime = "qwen"
	}
	if body.Runtime == "" {
		body.Runtime = "qwen"
	}
	if body.Runtime == "qwen" && body.Qwen.ModelPath == "" && body.Qwen.CodecPath == "" && body.ModelPath == "" && body.CodecPath == "" {
		body.ModelPath = os.Getenv("CHIRP_MODEL_PATH")
		body.CodecPath = os.Getenv("CHIRP_CODEC_PATH")
	}
	if body.Runtime == "kokoro" && body.Kokoro.ModelPath == "" && body.Kokoro.VoicesPath == "" {
		body.Kokoro.ModelPath = os.Getenv("CHIRP_KOKORO_MODEL_PATH")
		body.Kokoro.VoicesPath = os.Getenv("CHIRP_KOKORO_VOICES_PATH")
		body.Kokoro.EspeakDataPath = os.Getenv("CHIRP_ESPEAK_DATA_PATH")
	}

	params := LoadParams{Runtime: body.Runtime}
	switch body.Runtime {
	case "qwen":
		params.Qwen = chirpc.Params{
			ModelPath:   firstNonEmpty(body.Qwen.ModelPath, body.ModelPath),
			CodecPath:   firstNonEmpty(body.Qwen.CodecPath, body.CodecPath),
			MaxTokens:   firstNonZero(body.Qwen.MaxTokens, body.MaxTokens),
			Temperature: firstNonZeroFloat(body.Qwen.Temperature, body.Temperature),
			TopK:        firstNonZero(body.Qwen.TopK, body.TopK),
		}
		if params.Qwen.ModelPath == "" || params.Qwen.CodecPath == "" {
			writeError(w, http.StatusBadRequest, errInvalidRequest, "qwen runtime requires model_path and codec_path")
			return
		}
	case "kokoro":
		params.Kokoro = kokoroc.Params{
			ModelPath:      body.Kokoro.ModelPath,
			VoicesPath:     body.Kokoro.VoicesPath,
			EspeakDataPath: body.Kokoro.EspeakDataPath,
			Voice:          body.Kokoro.Voice,
			Language:       body.Kokoro.Language,
			Speed:          body.Kokoro.Speed,
		}
		if params.Kokoro.ModelPath == "" || params.Kokoro.VoicesPath == "" {
			writeError(w, http.StatusBadRequest, errInvalidRequest, "kokoro runtime requires model_path and voices_path")
			return
		}
	default:
		writeError(w, http.StatusBadRequest, errInvalidRequest, "unsupported runtime")
		return
	}
	if err := s.LoadModel(params); err != nil {
		writeError(w, http.StatusInternalServerError, errInternal, "failed to load model: "+err.Error())
		return
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]string{"status": "loaded", "model": s.modelName})
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if value != "" {
			return value
		}
	}
	return ""
}

func firstNonZero(values ...int) int {
	for _, value := range values {
		if value != 0 {
			return value
		}
	}
	return 0
}

func firstNonZeroFloat(values ...float32) float32 {
	for _, value := range values {
		if value != 0 {
			return value
		}
	}
	return 0
}

func (s *Server) handleModelUnload(w http.ResponseWriter, r *http.Request) {
	s.UnloadModel()
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]string{"status": "unloaded"})
}

func (s *Server) handleSpeech(w http.ResponseWriter, r *http.Request) {
	if !s.mu.TryLock() {
		writeError(w, http.StatusTooManyRequests, errBusy, "server is busy with another generation")
		return
	}
	defer s.mu.Unlock()

	if s.ctx == nil {
		writeError(w, http.StatusServiceUnavailable, errNoModel, "no model loaded")
		return
	}

	var body struct {
		Input          string `json:"input"`
		VoiceReference string `json:"voice_reference,omitempty"`
		Voice          string `json:"voice,omitempty"`
		ResponseFormat string `json:"response_format,omitempty"`
		Language       string `json:"language,omitempty"`
	}
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil || body.Input == "" {
		writeError(w, http.StatusBadRequest, errInvalidRequest, "request body must contain input")
		return
	}
	if body.ResponseFormat != "" && body.ResponseFormat != "wav" {
		writeError(w, http.StatusBadRequest, errInvalidRequest, "only wav response_format is supported")
		return
	}

	tmp, err := os.CreateTemp("", "chirp-speech-*.wav")
	if err != nil {
		writeError(w, http.StatusInternalServerError, errInternal, "failed to create temp output: "+err.Error())
		return
	}
	outPath := tmp.Name()
	tmp.Close()
	defer os.Remove(outPath)

	if err := s.ctx.SynthesizeToFile(body.Input, firstNonEmpty(body.VoiceReference, body.Voice), outPath, body.Language); err != nil {
		writeError(w, http.StatusInternalServerError, errInternal, err.Error())
		return
	}
	data, err := os.ReadFile(outPath)
	if err != nil {
		writeError(w, http.StatusInternalServerError, errInternal, "failed to read output WAV: "+err.Error())
		return
	}
	w.Header().Set("Content-Type", "audio/wav")
	w.Header().Set("Content-Disposition", `attachment; filename="speech.wav"`)
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(data)
}
