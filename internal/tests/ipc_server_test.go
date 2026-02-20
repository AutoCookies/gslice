package tests

import (
	"context"
	"gslice/internal/adapters/ipc"
	"gslice/internal/adapters/logging"
	"gslice/internal/adapters/metrics"
	"gslice/internal/adapters/store"
	"gslice/internal/application"
	"net"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestIPCServerRequestHandling(t *testing.T) {
	dir := t.TempDir()
	storePath := filepath.Join(dir, "state.json")
	sockPath := filepath.Join(dir, "ipc.sock")
	st, err := store.NewBoltStore(storePath)
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()
	svc := application.NewService(st, application.RealClock{}, metrics.New(false), logging.New())
	session, err := svc.AllocateSession(context.Background(), 100, time.Minute)
	if err != nil {
		t.Fatal(err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	server := ipc.NewUDSServer(sockPath, svc, logging.New(), "", false, 1000)
	if err := server.Start(ctx); err != nil {
		t.Fatal(err)
	}
	defer server.Stop()

	conn, err := net.Dial("unix", sockPath)
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close()

	resp := rpc(t, conn, ipc.Request{V: 1, Op: "reserve", SessionID: session.ID, PID: os.Getpid(), Bytes: 60})
	if !resp.OK || resp.UsedBytes != 60 || resp.RemainingBytes != 40 {
		t.Fatalf("bad reserve response: %#v", resp)
	}

	resp = rpc(t, conn, ipc.Request{V: 1, Op: "reserve", SessionID: session.ID, PID: os.Getpid(), Bytes: 50})
	if resp.OK || resp.Error == nil || resp.Error.Code != ipc.CodeQuotaExceeded || resp.UsedBytes != 60 {
		t.Fatalf("expected quota exceeded, got %#v", resp)
	}

	resp = rpc(t, conn, ipc.Request{V: 1, Op: "release", SessionID: session.ID, PID: os.Getpid(), Bytes: 30})
	if !resp.OK || resp.UsedBytes != 30 {
		t.Fatalf("bad release response: %#v", resp)
	}
	resp = rpc(t, conn, ipc.Request{V: 1, Op: "release", SessionID: session.ID, PID: os.Getpid(), Bytes: 100})
	if !resp.OK || resp.UsedBytes != 0 {
		t.Fatalf("release should clamp to zero: %#v", resp)
	}

	resp = rpc(t, conn, ipc.Request{V: 1, Op: "status", SessionID: session.ID, PID: os.Getpid(), Bytes: 0})
	if !resp.OK || resp.UsedBytes != 0 || resp.LimitBytes != 100 || resp.RemainingBytes != 100 {
		t.Fatalf("bad status response: %#v", resp)
	}

	resp = rpc(t, conn, ipc.Request{V: 1, Op: "status", SessionID: "missing", PID: os.Getpid(), Bytes: 0})
	if resp.OK || resp.Error == nil || resp.Error.Code != ipc.CodeNotFound {
		t.Fatalf("expected not found: %#v", resp)
	}
}

func rpc(t *testing.T, conn net.Conn, req ipc.Request) ipc.Response {
	t.Helper()
	if err := ipc.EncodeFrame(conn, req); err != nil {
		t.Fatal(err)
	}
	var resp ipc.Response
	if err := ipc.DecodeFrame(conn, &resp); err != nil {
		t.Fatal(err)
	}
	return resp
}
