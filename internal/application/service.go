package application

import (
	"context"
	"fmt"
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

func buildEnv(base []string, sessionID, sock, token, preload, libPath string) []string {
	env := append([]string{}, base...)
	env = append(env, "GPUSLICE_SESSION="+sessionID, "GPUSLICE_IPC_SOCK="+sock)
	if token != "" {
		env = append(env, "GPUSLICE_IPC_TOKEN="+token)
	}
	if preload != "" {
		env = append(env, "LD_PRELOAD="+preload)
	}
	if libPath != "" {
		env = append(env, "LD_LIBRARY_PATH="+libPath)
	}
	return env
}

func (s *Service) RunCommand(ctx context.Context, sessionID, sock, token, preload, libPath string, command []string) error {
	if len(command) == 0 {
		return fmt.Errorf("command required")
	}
	cmd := exec.CommandContext(ctx, command[0], command[1:]...)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	cmd.Stdin = os.Stdin
	cmd.Env = buildEnv(os.Environ(), sessionID, sock, token, preload, libPath)
	return cmd.Run()
}
