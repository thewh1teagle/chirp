package server

import (
	"encoding/json"
	"net/http"
)

type errorResponse struct {
	Error errorBody `json:"error"`
}

type errorBody struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

const (
	errInvalidRequest = "invalid_request"
	errNoModel        = "no_model"
	errBusy           = "busy"
	errInternal       = "internal_error"
)

func writeError(w http.ResponseWriter, status int, code, message string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(errorResponse{Error: errorBody{Code: code, Message: message}})
}
