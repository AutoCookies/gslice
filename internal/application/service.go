package application

import (
	"context"
	"fmt"
	"gslice/internal/domain"
	"gslice/internal/ports"
	"os"
	"os/exec"
	"time"
)

type RealClock struct{}

func (RealClock) Now() time.Time { return time.Now().UTC() }

type Service struct {
	store   ports.SessionStore
	clock   ports.Clock
	metrics ports.Metrics
	logger  ports.Logger
}

func NewService(store ports.SessionStore, clock ports.Clock, metrics ports.Metrics, logger ports.Logger) *Service {
	return &Service{store: store, clock: clock, metrics: metrics, logger: logger}
}

type ReserveResult struct {
	Allowed        bool   `json:"allowed"`
	UsedBytes      uint64 `json:"used_bytes"`
	RemainingBytes uint64 `json:"remaining_bytes"`
}

func (s *Service) Reserve(ctx context.Context, sessionID string, bytes uint64) (ReserveResult, error) {
	session, err := s.store.Get(ctx, sessionID)
	if err != nil {
		return ReserveResult{}, err
	}
	if session.IsExpired(s.clock.Now()) {
		return ReserveResult{}, domain.ErrSessionExpired
	}
	used, err := domain.Reserve(session.UsedBytes, session.VRAMLimitBytes, bytes)
	if err != nil {
		s.metrics.ObserveAllocationResult(false)
		return ReserveResult{Allowed: false, UsedBytes: session.UsedBytes, RemainingBytes: session.VRAMLimitBytes - session.UsedBytes}, nil
	}
	if err := s.store.UpdateUsedBytes(ctx, sessionID, used); err != nil {
		return ReserveResult{}, err
	}
	s.metrics.SetSessionUsage(sessionID, float64(used))
	s.metrics.ObserveAllocationResult(true)
	return ReserveResult{Allowed: true, UsedBytes: used, RemainingBytes: session.VRAMLimitBytes - used}, nil
}

func (s *Service) ReleaseBytes(ctx context.Context, sessionID string, bytes uint64) (ReserveResult, error) {
	session, err := s.store.Get(ctx, sessionID)
	if err != nil {
		return ReserveResult{}, err
	}
	used := domain.Release(session.UsedBytes, bytes)
	if err := s.store.UpdateUsedBytes(ctx, sessionID, used); err != nil {
		return ReserveResult{}, err
	}
	s.metrics.SetSessionUsage(sessionID, float64(used))
	return ReserveResult{Allowed: true, UsedBytes: used, RemainingBytes: session.VRAMLimitBytes - used}, nil
}

func buildEnv(base []string, sessionID, sock, preload, libPath string) []string {
	env := append([]string{}, base...)
	env = append(env, "GPUSLICE_SESSION="+sessionID, "GPUSLICE_IPC_SOCK="+sock)
	if preload != "" {
		env = append(env, "LD_PRELOAD="+preload)
	}
	if libPath != "" {
		env = append(env, "LD_LIBRARY_PATH="+libPath)
	}
	return env
}

func (s *Service) RunCommand(ctx context.Context, sessionID, sock, preload, libPath string, command []string) error {
	if len(command) == 0 {
		return fmt.Errorf("command required")
	}
	cmd := exec.CommandContext(ctx, command[0], command[1:]...)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	cmd.Stdin = os.Stdin
	cmd.Env = buildEnv(os.Environ(), sessionID, sock, preload, libPath)
	return cmd.Run()
}
