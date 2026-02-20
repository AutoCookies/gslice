package tests

import (
	"context"
	"gslice/internal/adapters/ipc"
	"gslice/internal/adapters/logging"
	"gslice/internal/adapters/metrics"
	"gslice/internal/adapters/store"
	"gslice/internal/application"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestMockModeEndToEnd(t *testing.T) {
	root := filepath.Join("..", "..")
	cmd := exec.Command("make", "build-native")
	cmd.Dir = root
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("build native failed: %v\n%s", err, string(out))
	}
	dir := t.TempDir()
	st, err := store.NewBoltStore(filepath.Join(dir, "test.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()
	svc := application.NewService(st, application.RealClock{}, metrics.New(), logging.New())
	sock := filepath.Join(dir, "ipc.sock")
	uds := ipc.NewUDSServer(sock, svc)
	ctx := context.Background()
	if err := uds.Start(ctx); err != nil {
		t.Fatal(err)
	}
	defer uds.Stop()
	session, err := svc.AllocateSession(ctx, 2*1024*1024, 10*time.Minute)
	if err != nil {
		t.Fatal(err)
	}
	target := exec.Command(filepath.Join(root, "examples", "target_app"), "4", "1048576")
	target.Env = append(target.Environ(),
		"GPUSLICE_SESSION="+session.ID,
		"GPUSLICE_SOCKET="+sock,
		"LD_PRELOAD="+filepath.Join(root, "native", "interceptor", "libgpuslice.so"),
		"LD_LIBRARY_PATH="+filepath.Join(root, "native", "fakecuda"),
	)
	out, err := target.CombinedOutput()
	if err != nil {
		t.Fatalf("target app failed: %v\n%s", err, string(out))
	}
	text := string(out)
	if !strings.Contains(text, "malloc_failed_at=2") {
		t.Fatalf("expected quota fail after 2 allocations, got: %s", text)
	}
	status, err := svc.GetStatus(ctx, session.ID)
	if err != nil {
		t.Fatal(err)
	}
	if status.UsedBytes != 0 {
		t.Fatalf("expected bytes released to zero, got %d", status.UsedBytes)
	}
}
