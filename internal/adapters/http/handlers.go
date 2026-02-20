package http

import (
	"encoding/json"
	"gslice/internal/application"
	"gslice/internal/domain"
	"net/http"
	"time"
)

type Handlers struct{ svc *application.Service }

func NewHandlers(svc *application.Service) Handlers { return Handlers{svc: svc} }

type allocateRequest struct {
	LimitBytes uint64 `json:"limit_bytes"`
	TTLSeconds int64  `json:"ttl_seconds"`
}

func (h Handlers) Allocate(w http.ResponseWriter, r *http.Request) {
	var req allocateRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil || req.LimitBytes == 0 || req.TTLSeconds <= 0 {
		http.Error(w, "invalid request", 400)
		return
	}
	s, err := h.svc.AllocateSession(r.Context(), req.LimitBytes, time.Duration(req.TTLSeconds)*time.Second)
	if err != nil {
		http.Error(w, err.Error(), 500)
		return
	}
	_ = json.NewEncoder(w).Encode(s)
}
func (h Handlers) List(w http.ResponseWriter, r *http.Request) {
	s, err := h.svc.ListSessions(r.Context())
	if err != nil {
		http.Error(w, err.Error(), 500)
		return
	}
	_ = json.NewEncoder(w).Encode(s)
}
func (h Handlers) Get(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	s, err := h.svc.GetStatus(r.Context(), id)
	if err != nil {
		if err == domain.ErrSessionNotFound {
			http.Error(w, err.Error(), 404)
			return
		}
		http.Error(w, err.Error(), 500)
		return
	}
	_ = json.NewEncoder(w).Encode(s)
}
func (h Handlers) Release(w http.ResponseWriter, r *http.Request) {
	if err := h.svc.ReleaseSession(r.Context(), r.PathValue("id")); err != nil {
		http.Error(w, err.Error(), 500)
		return
	}
	w.WriteHeader(204)
}
