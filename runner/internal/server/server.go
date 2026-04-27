package server

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"sync"
	"syscall"
	"time"

	"github.com/thewh1teagle/chirp/runner/internal/chirpc"
)

type Server struct {
	mu        sync.Mutex
	ctx       *chirpc.Context
	modelName string
	modelPath string
	codecPath string
	Version   string
	Commit    string
}

func New() *Server {
	return &Server{}
}

func (s *Server) LoadModel(params chirpc.Params) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.loadModelLocked(params)
}

func (s *Server) loadModelLocked(params chirpc.Params) error {
	if s.ctx != nil {
		s.ctx.Close()
		s.ctx = nil
	}
	ctx, err := chirpc.New(params)
	if err != nil {
		return err
	}
	s.ctx = ctx
	s.modelPath = params.ModelPath
	s.codecPath = params.CodecPath
	s.modelName = filepath.Base(params.ModelPath)
	return nil
}

func (s *Server) UnloadModel() {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.ctx != nil {
		s.ctx.Close()
		s.ctx = nil
	}
	s.modelName = ""
	s.modelPath = ""
	s.codecPath = ""
}

func (s *Server) Languages() []chirpc.Language {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.ctx == nil {
		return nil
	}
	return s.ctx.Languages()
}

func (s *Server) Close() {
	s.UnloadModel()
}

func recoveryMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer func() {
			if err := recover(); err != nil {
				log.Printf("panic recovered: %v", err)
				writeError(w, http.StatusInternalServerError, errInternal, fmt.Sprintf("internal error: %v", err))
			}
		}()
		next.ServeHTTP(w, r)
	})
}

func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /health", s.handleHealth)
	mux.HandleFunc("GET /ready", s.handleReady)
	mux.HandleFunc("GET /v1/languages", s.handleLanguages)
	mux.HandleFunc("GET /v1/models", s.handleModels)
	mux.HandleFunc("POST /v1/models/load", s.handleModelLoad)
	mux.HandleFunc("DELETE /v1/models", s.handleModelUnload)
	mux.HandleFunc("POST /v1/audio/speech", s.handleSpeech)
	return recoveryMiddleware(mux)
}

func ListenAndServe(host string, port int, s *Server) error {
	ln, err := net.Listen("tcp", fmt.Sprintf("%s:%d", host, port))
	if err != nil {
		return err
	}
	actualPort := ln.Addr().(*net.TCPAddr).Port
	readyMsg, _ := json.Marshal(map[string]any{
		"status":  "ready",
		"port":    actualPort,
		"version": s.Version,
		"commit":  s.Commit,
	})
	fmt.Println(string(readyMsg))
	log.Printf("listening on %s:%d", host, actualPort)

	srv := &http.Server{Handler: s.Handler()}
	go func() {
		sigCh := make(chan os.Signal, 1)
		signal.Notify(sigCh, os.Interrupt, syscall.SIGTERM)
		<-sigCh
		log.Println("shutting down...")
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		_ = srv.Shutdown(ctx)
		s.Close()
	}()

	if err := srv.Serve(ln); err != http.ErrServerClosed {
		return err
	}
	return nil
}
