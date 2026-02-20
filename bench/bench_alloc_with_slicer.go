//go:build benchtool

package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"gslice/internal/adapters/ipc"
	"gslice/internal/adapters/logging"
	"gslice/internal/adapters/metrics"
	"gslice/internal/adapters/store"
	"gslice/internal/application"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

type allocResult struct {
	Name       string  `json:"name"`
	Rounds     int     `json:"rounds"`
	Iterations int     `json:"iterations"`
	Bytes      int     `json:"bytes"`
	MeanNSOp   float64 `json:"mean_ns_op"`
	P50NSOp    float64 `json:"p50_ns_op"`
	P95NSOp    float64 `json:"p95_ns_op"`
	TotalSec   float64 `json:"total_sec"`
	AllocsPerS float64 `json:"allocs_per_sec"`
}

func percentile(v []float64, p float64) float64 { sort.Float64s(v); return v[int(float64(len(v)-1)*p)] }

func main() {
	rounds := flag.Int("rounds", 20, "rounds")
	iters := flag.Int("iterations", 2000, "iterations")
	bytes := flag.Int("bytes", 4096, "bytes")
	jsonOut := flag.Bool("json", false, "json")
	flag.Parse()

	dir, _ := os.MkdirTemp("", "bench-slicer-")
	defer os.RemoveAll(dir)
	st, _ := store.NewBoltStore(filepath.Join(dir, "state.json"))
	svc := application.NewService(st, application.RealClock{}, metrics.New(false), logging.New())
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	svc.StartBackground(ctx, 100*time.Millisecond)
	sock := filepath.Join(dir, "ipc.sock")
	token := "bench-token"
	uds := ipc.NewUDSServer(sock, svc, logging.New(), token, true, 5000)
	if err := uds.Start(ctx); err != nil {
		panic(err)
	}
	defer uds.Stop()
	session, err := svc.AllocateSession(context.Background(), uint64(1<<30), 10*time.Minute)
	if err != nil {
		panic(err)
	}

	nsOps := make([]float64, 0, *rounds)
	var total time.Duration
	for i := 0; i < *rounds; i++ {
		cmd := exec.Command("../native/examples/target_app")
		cmd.Env = append(os.Environ(),
			"FAKECUDA_TOTAL_MEM=1073741824",
			fmt.Sprintf("ALLOC_BYTES=%d", *bytes),
			fmt.Sprintf("ITERATIONS=%d", *iters),
			"GPUSLICE_SESSION="+session.ID,
			"GPUSLICE_IPC_SOCK="+sock,
			"GPUSLICE_IPC_TOKEN="+token,
			"LD_PRELOAD="+abs("../native/interceptor/libgpuslice.so"),
		)
		start := time.Now()
		out, err := cmd.CombinedOutput()
		d := time.Since(start)
		if err != nil {
			panic(fmt.Sprintf("with-slicer run failed: %v\n%s", err, string(out)))
		}
		if strings.Count(string(out), "ALLOC_OK") != *iters {
			panic("allocation count mismatch with slicer: " + string(out))
		}
		total += d
		nsOps = append(nsOps, float64(d.Nanoseconds())/float64(*iters))
	}
	res := allocResult{Name: "with_slicer", Rounds: *rounds, Iterations: *iters, Bytes: *bytes, MeanNSOp: float64(total.Nanoseconds()) / float64(*rounds**iters), P50NSOp: percentile(nsOps, 0.50), P95NSOp: percentile(nsOps, 0.95), TotalSec: total.Seconds(), AllocsPerS: float64(*rounds**iters) / total.Seconds()}
	if *jsonOut {
		_ = json.NewEncoder(os.Stdout).Encode(res)
		return
	}
	fmt.Printf("%s: mean %.2f ns/op p50 %.2f p95 %.2f total %.3fs alloc/s %.0f\n", res.Name, res.MeanNSOp, res.P50NSOp, res.P95NSOp, res.TotalSec, res.AllocsPerS)
}

func abs(p string) string { a, _ := filepath.Abs(p); return a }
