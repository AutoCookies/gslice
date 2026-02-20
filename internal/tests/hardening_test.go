package tests

import (
	"context"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	httpadapter "gslice/internal/adapters/http"
	"gslice/internal/adapters/ipc"
	"gslice/internal/adapters/logging"
	"gslice/internal/adapters/metrics"
	"gslice/internal/adapters/store"
	"gslice/internal/application"
)

func TestCrashRecovery(t *testing.T) {
	root := repoRoot(t)
	mustBuildNative(t, root)
	svc, sock, stop := startServerWith(t, "tok", true, 50*time.Millisecond)
	defer stop()
	sess, _ := svc.AllocateSession(context.Background(), 128*1024*1024, time.Minute)
	_ = runBin(t, filepath.Join(root, "native", "examples", "crash_alloc"), map[string]string{
		"GPUSLICE_SESSION": sess.ID, "GPUSLICE_IPC_SOCK": sock, "GPUSLICE_IPC_TOKEN": "tok", "LD_PRELOAD": filepath.Join(root, "native", "interceptor", "libgpuslice.so"), "ALLOC_BYTES": "67108864", "ITERATIONS": "2", "FAKECUDA_TOTAL_MEM": "1073741824",
	})
	waitUntil(t, 2*time.Second, func() bool { s, _ := svc.GetStatus(context.Background(), sess.ID); return s.UsedBytes == 0 })
}

func TestServerRestartRecovery(t *testing.T) {
	root := repoRoot(t)
	mustBuildNative(t, root)
	dir := t.TempDir()
	st, _ := store.NewBoltStore(filepath.Join(dir, "state.json"))
	m := metrics.New(false)
	svc := application.NewService(st, application.RealClock{}, m, logging.New())
	sess, _ := svc.AllocateSession(context.Background(), 128*1024*1024, time.Minute)
	sock := filepath.Join(dir, "ipc.sock")
	ctx1, cancel1 := context.WithCancel(context.Background())
	s1 := ipc.NewUDSServer(sock, svc, logging.New(), "tok", true, 1000)
	_ = s1.Start(ctx1)
	waitForUDSReady(t, sock, 300*time.Millisecond)
	_ = runBin(t, filepath.Join(root, "native", "examples", "crash_alloc"), map[string]string{"GPUSLICE_SESSION": sess.ID, "GPUSLICE_IPC_SOCK": sock, "GPUSLICE_IPC_TOKEN": "tok", "LD_PRELOAD": filepath.Join(root, "native", "interceptor", "libgpuslice.so"), "ALLOC_BYTES": "67108864", "ITERATIONS": "1", "FAKECUDA_TOTAL_MEM": "1073741824"})
	cancel1()
	_ = s1.Stop()
	_ = st.Close()

	st2, _ := store.NewBoltStore(filepath.Join(dir, "state.json"))
	svc2 := application.NewService(st2, application.RealClock{}, metrics.New(false), logging.New())
	ctx2, cancel2 := context.WithCancel(context.Background())
	defer cancel2()
	svc2.StartBackground(ctx2, 50*time.Millisecond)
	s2 := ipc.NewUDSServer(sock, svc2, logging.New(), "tok", true, 1000)
	_ = s2.Start(ctx2)
	defer func() { _ = s2.Stop(); _ = st2.Close() }()
	waitUntil(t, 2*time.Second, func() bool { s, _ := svc2.GetStatus(context.Background(), sess.ID); return s.UsedBytes == 0 })
}

func TestUnauthorizedIPC(t *testing.T) {
	root := repoRoot(t)
	mustBuildNative(t, root)
	svc, sock, stop := startServerWith(t, "right", true, 100*time.Millisecond)
	defer stop()
	sess, _ := svc.AllocateSession(context.Background(), 128*1024*1024, time.Minute)
	out := runBin(t, filepath.Join(root, "native", "examples", "target_app"), map[string]string{"GPUSLICE_SESSION": sess.ID, "GPUSLICE_IPC_SOCK": sock, "GPUSLICE_IPC_TOKEN": "wrong", "LD_PRELOAD": filepath.Join(root, "native", "interceptor", "libgpuslice.so"), "ALLOC_BYTES": "67108864", "ITERATIONS": "2", "FAKECUDA_TOTAL_MEM": "1073741824"})
	if !strings.Contains(out, "ALLOC_FAIL at iteration 0 code=2") {
		t.Fatalf("unexpected output: %s", out)
	}
}

func TestTTLExpiration(t *testing.T) {
	svc, sock, stop := startServerWith(t, "tok", true, 50*time.Millisecond)
	defer stop()
	sess, _ := svc.AllocateSession(context.Background(), 128*1024*1024, 500*time.Millisecond)
	waitUntil(t, 2*time.Second, func() bool {
		c, err := net.Dial("unix", sock)
		if err != nil {
			return false
		}
		defer c.Close()
		_ = ipc.EncodeFrame(c, ipc.Request{V: 1, Op: "reserve", SessionID: sess.ID, PID: os.Getpid(), Bytes: 1, Token: "tok"})
		var resp ipc.Response
		_ = ipc.DecodeFrame(c, &resp)
		return !resp.OK && resp.Error != nil && resp.Error.Code == ipc.CodeExpired
	})
}

func TestMetricsCardinalitySafe(t *testing.T) {
	st, _ := store.NewBoltStore(filepath.Join(t.TempDir(), "s.json"))
	m := metrics.New(false)
	svc := application.NewService(st, application.RealClock{}, m, logging.New())
	h := httpadapter.NewHandlers(svc)
	r := httptest.NewServer(httpadapter.NewRouter(h, m))
	defer r.Close()
	res, _ := http.Get(r.URL + "/metrics")
	defer res.Body.Close()
	body, _ := io.ReadAll(res.Body)
	text := string(body)
	for _, k := range []string{"gpuslice_sessions_active", "gpuslice_used_bytes_total", "gpuslice_alloc_events_total", "gpuslice_denied_alloc_total", "gpuslice_recovered_bytes_total"} {
		if !strings.Contains(text, k) {
			t.Fatalf("missing metric %s", k)
		}
	}
	if strings.Contains(text, "session_id=") {
		t.Fatalf("unexpected high-cardinality label in default metrics: %s", text)
	}
}

func startServerWith(t *testing.T, token string, required bool, interval time.Duration) (*application.Service, string, func()) {
	t.Helper()
	dir := t.TempDir()
	st, _ := store.NewBoltStore(filepath.Join(dir, "state.json"))
	svc := application.NewService(st, application.RealClock{}, metrics.New(false), logging.New())
	ctx, cancel := context.WithCancel(context.Background())
	svc.StartBackground(ctx, interval)
	sock := filepath.Join(dir, "ipc.sock")
	server := ipc.NewUDSServer(sock, svc, logging.New(), token, required, 1000)
	if err := server.Start(ctx); err != nil {
		t.Fatal(err)
	}
	waitForUDSReady(t, sock, 300*time.Millisecond)
	return svc, sock, func() { cancel(); _ = server.Stop(); _ = st.Close() }
}

func runBin(t *testing.T, bin string, env map[string]string) string {
	t.Helper()
	cmd := exec.Command(bin)
	e := os.Environ()
	for k, v := range env {
		e = append(e, k+"="+v)
	}
	cmd.Env = e
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("run %s failed: %v\n%s", bin, err, string(out))
	}
	return string(out)
}

func waitUntil(t *testing.T, timeout time.Duration, fn func() bool) {
	t.Helper()
	d := time.Now().Add(timeout)
	for time.Now().Before(d) {
		if fn() {
			return
		}
		time.Sleep(20 * time.Millisecond)
	}
	t.Fatalf("condition timeout after %s", timeout)
}
