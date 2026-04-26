package server

import (
	"encoding/json"
	"net/http"
	"os"

	"github.com/thewh1teagle/chirp/runner/internal/chirpc"
)

func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

func (s *Server) handleReady(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	s.mu.Lock()
	loaded := s.ctx != nil
	name := s.modelName
	s.mu.Unlock()
	if !loaded {
		w.WriteHeader(http.StatusServiceUnavailable)
		_ = json.NewEncoder(w).Encode(map[string]string{"status": "not_ready", "message": "no model loaded"})
		return
	}
	_ = json.NewEncoder(w).Encode(map[string]string{"status": "ready", "model": name})
}

func (s *Server) handleModels(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	s.mu.Lock()
	defer s.mu.Unlock()
	_ = json.NewEncoder(w).Encode(map[string]any{
		"loaded": s.ctx != nil,
		"model":  s.modelName,
		"path":   s.modelPath,
		"codec":  s.codecPath,
	})
}

func (s *Server) handleModelLoad(w http.ResponseWriter, r *http.Request) {
	var body struct {
		ModelPath   string  `json:"model_path"`
		CodecPath   string  `json:"codec_path"`
		MaxTokens   int     `json:"max_tokens,omitempty"`
		Temperature float32 `json:"temperature,omitempty"`
		TopK        int     `json:"top_k,omitempty"`
	}
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil || body.ModelPath == "" || body.CodecPath == "" {
		writeError(w, http.StatusBadRequest, errInvalidRequest, "request body must contain model_path and codec_path")
		return
	}
	if err := s.LoadModel(chirpc.Params{
		ModelPath:   body.ModelPath,
		CodecPath:   body.CodecPath,
		MaxTokens:   body.MaxTokens,
		Temperature: body.Temperature,
		TopK:        body.TopK,
	}); err != nil {
		writeError(w, http.StatusInternalServerError, errInternal, "failed to load model: "+err.Error())
		return
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]string{"status": "loaded", "model": s.modelName})
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
		ResponseFormat string `json:"response_format,omitempty"`
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

	if err := s.ctx.SynthesizeToFile(body.Input, body.VoiceReference, outPath); err != nil {
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
