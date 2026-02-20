//go:build benchtool

package main

import (
	"context"
	"encoding/json"
	"flag"
	"gslice/internal/adapters/ipc"
	"gslice/internal/adapters/logging"
	"gslice/internal/adapters/metrics"
	"gslice/internal/adapters/store"
	"gslice/internal/application"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"
)

type stressResult struct {
	Processes      int     `json:"processes"`
	IterationsEach int     `json:"iterations_each"`
	TotalSec       float64 `json:"total_sec"`
	Success        int     `json:"success"`
	Denied         int     `json:"denied"`
	RecoveryOK     bool    `json:"recovery_ok"`
}

func main() {
	procs := flag.Int("procs", 10, "processes")
	iters := flag.Int("iters", 100, "iterations each")
	jsonOut := flag.Bool("json", false, "json")
	flag.Parse()

	dir, _ := os.MkdirTemp("", "bench-stress-")
	defer os.RemoveAll(dir)
	st, _ := store.NewBoltStore(filepath.Join(dir, "state.json"))
	svc := application.NewService(st, application.RealClock{}, metrics.New(false), logging.New())
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	svc.StartBackground(ctx, 100*time.Millisecond)
	sock := filepath.Join(dir, "ipc.sock")
	token := "bench-token"
	uds := ipc.NewUDSServer(sock, svc, logging.New(), token, true, 10000)
	_ = uds.Start(ctx)
	defer uds.Stop()

	start := time.Now()
	var wg sync.WaitGroup
	var mu sync.Mutex
	totalOK, totalDenied := 0, 0
	sessionIDs := make([]string, 0, *procs)

	for i := 0; i < *procs; i++ {
		s, _ := svc.AllocateSession(context.Background(), 1*1024*1024, 5*time.Minute)
		sessionIDs = append(sessionIDs, s.ID)
		wg.Add(1)
		go func(session string) {
			defer wg.Done()
			cmd := exec.Command("../native/examples/target_app")
			cmd.Env = append(os.Environ(),
				"FAKECUDA_TOTAL_MEM=1073741824",
				"ALLOC_BYTES=1048576",
				"ITERATIONS="+itoa(*iters),
				"GPUSLICE_SESSION="+session,
				"GPUSLICE_IPC_SOCK="+sock,
				"GPUSLICE_IPC_TOKEN="+token,
				"LD_PRELOAD="+abs("../native/interceptor/libgpuslice.so"),
			)
			out, _ := cmd.CombinedOutput()
			txt := string(out)
			ok := strings.Count(txt, "ALLOC_OK")
			denied := 0
			if strings.Contains(txt, "ALLOC_FAIL") {
				denied = 1
			}
			mu.Lock()
			totalOK += ok
			totalDenied += denied
			mu.Unlock()
		}(s.ID)
	}
	wg.Wait()
	elapsed := time.Since(start)

	recoveryOK := true
	for _, sid := range sessionIDs {
		stt, _ := svc.GetStatus(context.Background(), sid)
		if stt.UsedBytes != 0 {
			recoveryOK = false
			break
		}
	}

	res := stressResult{Processes: *procs, IterationsEach: *iters, TotalSec: elapsed.Seconds(), Success: totalOK, Denied: totalDenied, RecoveryOK: recoveryOK}
	if *jsonOut {
		_ = json.NewEncoder(os.Stdout).Encode(res)
		return
	}
	_ = json.NewEncoder(os.Stdout).Encode(res)
}

func abs(p string) string { a, _ := filepath.Abs(p); return a }
func itoa(v int) string   { return strconv.Itoa(v) }
