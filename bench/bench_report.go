//go:build benchtool

package main

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
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

type ipcOut struct {
	Sequential map[string]any `json:"sequential"`
	Concurrent map[string]any `json:"concurrent"`
}

type stressResult struct {
	Processes      int     `json:"processes"`
	IterationsEach int     `json:"iterations_each"`
	TotalSec       float64 `json:"total_sec"`
	Success        int     `json:"success"`
	Denied         int     `json:"denied"`
	RecoveryOK     bool    `json:"recovery_ok"`
}

type report struct {
	Baseline   allocResult  `json:"baseline"`
	WithSlicer allocResult  `json:"with_slicer"`
	IPC        ipcOut       `json:"ipc"`
	Stress     stressResult `json:"stress"`
}

func runJSON(path string, args ...string) []byte {
	cmd := exec.Command("go", append([]string{"run", "-tags", "benchtool", path}, args...)...)
	out, err := cmd.CombinedOutput()
	if err != nil {
		panic(fmt.Sprintf("run %s failed: %v\n%s", path, err, string(out)))
	}
	return out
}

func main() {
	var r report
	_ = json.Unmarshal(runJSON("./bench_alloc_baseline.go", "-json", "-rounds", "10", "-iterations", "200", "-bytes", "4096"), &r.Baseline)
	_ = json.Unmarshal(runJSON("./bench_alloc_with_slicer.go", "-json", "-rounds", "10", "-iterations", "200", "-bytes", "4096"), &r.WithSlicer)
	_ = json.Unmarshal(runJSON("./bench_ipc_latency.go", "-json"), &r.IPC)
	_ = json.Unmarshal(runJSON("./bench_stress.go", "-json"), &r.Stress)

	overhead := ((r.WithSlicer.MeanNSOp - r.Baseline.MeanNSOp) / r.Baseline.MeanNSOp) * 100
	fmt.Printf("Baseline alloc avg: %.2f ns/op\n", r.Baseline.MeanNSOp)
	fmt.Printf("With slicer alloc avg: %.2f ns/op (overhead %+0.2f%%)\n", r.WithSlicer.MeanNSOp, overhead)
	fmt.Printf("IPC mean us: seq %.2f, conc %.2f\n", r.IPC.Sequential["mean_us"], r.IPC.Concurrent["mean_us"])
	fmt.Printf("Stress: success=%d denied=%d recovery_ok=%v total=%.3fs\n", r.Stress.Success, r.Stress.Denied, r.Stress.RecoveryOK, r.Stress.TotalSec)

	_ = os.MkdirAll("bench", 0o755)
	f, _ := os.Create("results.json")
	defer f.Close()
	enc := json.NewEncoder(f)
	enc.SetIndent("", "  ")
	_ = enc.Encode(r)
}
