//go:build benchtool

package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"os/exec"
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

func percentile(v []float64, p float64) float64 {
	if len(v) == 0 {
		return 0
	}
	sort.Float64s(v)
	i := int(float64(len(v)-1) * p)
	return v[i]
}

func main() {
	rounds := flag.Int("rounds", 20, "number of rounds")
	iters := flag.Int("iterations", 2000, "allocations per round")
	bytes := flag.Int("bytes", 4096, "allocation size")
	jsonOut := flag.Bool("json", false, "json output")
	flag.Parse()

	nsOps := make([]float64, 0, *rounds)
	var total time.Duration

	for i := 0; i < *rounds; i++ {
		cmd := exec.Command("../native/examples/target_app")
		cmd.Env = append(os.Environ(),
			"FAKECUDA_TOTAL_MEM=1073741824",
			fmt.Sprintf("ALLOC_BYTES=%d", *bytes),
			fmt.Sprintf("ITERATIONS=%d", *iters),
		)
		start := time.Now()
		out, err := cmd.CombinedOutput()
		d := time.Since(start)
		if err != nil {
			panic(fmt.Sprintf("baseline run failed: %v\n%s", err, string(out)))
		}
		if strings.Count(string(out), "ALLOC_OK") != *iters {
			panic("allocation count mismatch")
		}
		total += d
		nsOps = append(nsOps, float64(d.Nanoseconds())/float64(*iters))
	}

	res := allocResult{
		Name:       "baseline_fakecuda",
		Rounds:     *rounds,
		Iterations: *iters,
		Bytes:      *bytes,
		MeanNSOp:   float64(total.Nanoseconds()) / float64(*rounds**iters),
		P50NSOp:    percentile(nsOps, 0.50),
		P95NSOp:    percentile(nsOps, 0.95),
		TotalSec:   total.Seconds(),
		AllocsPerS: float64(*rounds**iters) / total.Seconds(),
	}
	if *jsonOut {
		_ = json.NewEncoder(os.Stdout).Encode(res)
		return
	}
	fmt.Printf("%s: mean %.2f ns/op p50 %.2f p95 %.2f total %.3fs alloc/s %.0f\n", res.Name, res.MeanNSOp, res.P50NSOp, res.P95NSOp, res.TotalSec, res.AllocsPerS)
}
