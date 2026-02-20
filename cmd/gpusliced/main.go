package main

import (
	"context"
	httpadapter "gslice/internal/adapters/http"
	"gslice/internal/adapters/ipc"
	"gslice/internal/adapters/logging"
	"gslice/internal/adapters/metrics"
	"gslice/internal/adapters/store"
	"gslice/internal/application"
	"gslice/internal/config"
	"net/http"
	"os/signal"
	"syscall"
)

func main() {
	cfg, err := config.Load()
	if err != nil {
		panic(err)
	}
	logger := logging.New()
	st, err := store.NewBoltStore(cfg.DBPath)
	if err != nil {
		panic(err)
	}
	defer st.Close()
	metric := metrics.New(cfg.MetricsDebug)
	svc := application.NewService(st, application.RealClock{}, metric, logger)
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()
	svc.StartBackground(ctx, cfg.RecoveryInterval)
	uds := ipc.NewUDSServer(cfg.IPCSocketPath, svc, logger, cfg.IPCToken, cfg.RequireIPCToken, cfg.IPCRatePerSecond)
	if err := uds.Start(ctx); err != nil {
		panic(err)
	}
	defer uds.Stop()
	h := httpadapter.NewHandlers(svc)
	server := &http.Server{Addr: cfg.HTTPAddr, Handler: httpadapter.NewRouter(h, metric)}
	go func() { <-ctx.Done(); _ = server.Shutdown(context.Background()) }()
	logger.Info("server_start", "http_addr", cfg.HTTPAddr, "socket", cfg.IPCSocketPath)
	if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		panic(err)
	}
}
