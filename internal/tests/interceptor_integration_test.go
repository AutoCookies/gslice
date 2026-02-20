package tests

import (
	"context"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"gslice/internal/adapters/ipc"
	"gslice/internal/adapters/logging"
	"gslice/internal/adapters/metrics"
	"gslice/internal/adapters/store"
	"gslice/internal/application"
)

func TestInterceptorEnforcesQuota(t *testing.T) {
	root := repoRoot(t)
	mustBuildNative(t, root)
	svc, sock, stop := startServer(t)
	defer stop()

	session, err := svc.AllocateSession(context.Background(), 128*1024*1024, time.Minute)
	if err != nil {
		t.Fatal(err)
	}
	out := runTargetApp(t, root, runConfig{
		Socket:      sock,
		SessionID:   session.ID,
		TotalMem:    "1073741824",
		AllocBytes:  "67108864",
		Iterations:  "5",
		WithSession: true,
	})
	if strings.Count(out, "ALLOC_OK") != 2 || !strings.Contains(out, "ALLOC_FAIL at iteration 2 code=2") {
		t.Fatalf("unexpected output:\n%s", out)
	}
	assertUsedBytes(t, svc, session.ID, 0)
}

func TestInterceptorNoSessionNoEnforcement(t *testing.T) {
	root := repoRoot(t)
	mustBuildNative(t, root)
	_, sock, stop := startServer(t)
	defer stop()

	out := runTargetApp(t, root, runConfig{
		Socket:      sock,
		TotalMem:    "134217728",
		AllocBytes:  "67108864",
		Iterations:  "5",
		WithSession: false,
	})
	if strings.Count(out, "ALLOC_OK") != 2 || !strings.Contains(out, "ALLOC_FAIL at iteration 2 code=2") {
		t.Fatalf("unexpected output:\n%s", out)
	}
}

func TestInterceptorFailClosedWhenIPCDown(t *testing.T) {
	root := repoRoot(t)
	mustBuildNative(t, root)

	out := runTargetApp(t, root, runConfig{
		Socket:      filepath.Join(t.TempDir(), "missing.sock"),
		SessionID:   "sess_ok",
		TotalMem:    "1073741824",
		AllocBytes:  "67108864",
		Iterations:  "5",
		WithSession: true,
	})
	if !strings.Contains(out, "ALLOC_FAIL at iteration 0 code=2") {
		t.Fatalf("expected immediate failure, got:\n%s", out)
	}
}

func TestInterceptorRollbackReservationWhenUnderlyingFails(t *testing.T) {
	root := repoRoot(t)
	mustBuildNative(t, root)
	svc, sock, stop := startServer(t)
	defer stop()

	session, err := svc.AllocateSession(context.Background(), 512*1024*1024, time.Minute)
	if err != nil {
		t.Fatal(err)
	}
	out := runTargetApp(t, root, runConfig{
		Socket:      sock,
		SessionID:   session.ID,
		TotalMem:    "67108864",
		AllocBytes:  "134217728",
		Iterations:  "1",
		WithSession: true,
	})
	if !strings.Contains(out, "ALLOC_FAIL at iteration 0 code=2") {
		t.Fatalf("unexpected output:\n%s", out)
	}
	assertUsedBytes(t, svc, session.ID, 0)
}

func TestInterceptorFreeReleasesQuota(t *testing.T) {
	root := repoRoot(t)
	mustBuildNative(t, root)
	svc, sock, stop := startServer(t)
	defer stop()

	session, err := svc.AllocateSession(context.Background(), 128*1024*1024, time.Minute)
	if err != nil {
		t.Fatal(err)
	}
	out := runTargetApp(t, root, runConfig{
		Socket:      sock,
		SessionID:   session.ID,
		TotalMem:    "1073741824",
		AllocBytes:  "67108864",
		Iterations:  "2",
		WithSession: true,
	})
	if strings.Count(out, "ALLOC_OK") != 2 || strings.Contains(out, "ALLOC_FAIL") || strings.Count(out, "FREE_OK") != 2 {
		t.Fatalf("unexpected output:\n%s", out)
	}
	assertUsedBytes(t, svc, session.ID, 0)
}

type runConfig struct {
	Socket      string
	SessionID   string
	TotalMem    string
	AllocBytes  string
	Iterations  string
	WithSession bool
}

func runTargetApp(t *testing.T, root string, cfg runConfig) string {
	t.Helper()
	cmd := exec.Command(filepath.Join(root, "native", "examples", "target_app"))
	env := append(os.Environ(),
		"FAKECUDA_TOTAL_MEM="+cfg.TotalMem,
		"ALLOC_BYTES="+cfg.AllocBytes,
		"ITERATIONS="+cfg.Iterations,
		"GPUSLICE_IPC_SOCK="+cfg.Socket,
		"LD_PRELOAD="+filepath.Join(root, "native", "interceptor", "libgpuslice.so"),
	)
	if cfg.WithSession {
		env = append(env, "GPUSLICE_SESSION="+cfg.SessionID)
	}
	cmd.Env = env
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("target_app failed: %v\n%s", err, string(out))
	}
	return string(out)
}

func startServer(t *testing.T) (*application.Service, string, func()) {
	t.Helper()
	dir := t.TempDir()
	st, err := store.NewBoltStore(filepath.Join(dir, "state.json"))
	if err != nil {
		t.Fatal(err)
	}
	svc := application.NewService(st, application.RealClock{}, metrics.New(), logging.New())
	sock := filepath.Join(dir, "ipc.sock")
	ctx, cancel := context.WithCancel(context.Background())
	server := ipc.NewUDSServer(sock, svc, logging.New())
	if err := server.Start(ctx); err != nil {
		t.Fatal(err)
	}
	waitForUDSReady(t, sock, 200*time.Millisecond)
	return svc, sock, func() {
		cancel()
		_ = server.Stop()
		_ = st.Close()
	}
}

func waitForUDSReady(t *testing.T, sock string, timeout time.Duration) {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		conn, err := net.DialTimeout("unix", sock, 20*time.Millisecond)
		if err == nil {
			_ = conn.Close()
			return
		}
	}
	t.Fatalf("uds not ready: %s", sock)
}

func assertUsedBytes(t *testing.T, svc *application.Service, sessionID string, want uint64) {
	t.Helper()
	status, err := svc.GetStatus(context.Background(), sessionID)
	if err != nil {
		t.Fatal(err)
	}
	if status.UsedBytes != want {
		t.Fatalf("used bytes mismatch: got %d want %d", status.UsedBytes, want)
	}
}

func repoRoot(t *testing.T) string {
	t.Helper()
	return filepath.Join("..", "..")
}

func mustBuildNative(t *testing.T, root string) {
	t.Helper()
	cmd := exec.Command("make", "build-native")
	cmd.Dir = root
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("build-native failed: %v\n%s", err, string(out))
	}
}
