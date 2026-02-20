package http

import "net/http"

type metricsRenderer interface{ RenderPrometheus() string }

func NewRouter(h Handlers, m metricsRenderer) http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("POST /sessions", h.Allocate)
	mux.HandleFunc("GET /sessions", h.List)
	mux.HandleFunc("GET /sessions/{id}", h.Get)
	mux.HandleFunc("DELETE /sessions/{id}", h.Release)
	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("ok"))
	})
	mux.HandleFunc("GET /readyz", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("ready"))
	})
	mux.HandleFunc("GET /metrics", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/plain; version=0.0.4")
		_, _ = w.Write([]byte(m.RenderPrometheus()))
	})
	return mux
}
