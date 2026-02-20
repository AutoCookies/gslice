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
	"net"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"time"
)

type latResult struct {
	Name   string  `json:"name"`
	N      int     `json:"n"`
	MeanUS float64 `json:"mean_us"`
	P50US  float64 `json:"p50_us"`
	P95US  float64 `json:"p95_us"`
	P99US  float64 `json:"p99_us"`
}

type out struct {
	Sequential latResult `json:"sequential"`
	Concurrent latResult `json:"concurrent"`
}

func main() {
	n := flag.Int("n", 10000, "calls")
	jsonOut := flag.Bool("json", false, "json")
	flag.Parse()

	dir, _ := os.MkdirTemp("", "bench-ipc-")
	defer os.RemoveAll(dir)
	st, _ := store.NewBoltStore(filepath.Join(dir, "state.json"))
	svc := application.NewService(st, application.RealClock{}, metrics.New(false), logging.New())
	sock := filepath.Join(dir, "ipc.sock")
	token := "bench-token"
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	uds := ipc.NewUDSServer(sock, svc, logging.New(), token, true, 100000)
	_ = uds.Start(ctx)
	defer uds.Stop()
	sess, _ := svc.AllocateSession(context.Background(), 1<<30, 10*time.Minute)

	seq := runSeq(sock, token, sess.ID, *n)
	con := runCon(sock, token, sess.ID, *n, 32)
	r := out{Sequential: seq, Concurrent: con}
	if *jsonOut {
		_ = json.NewEncoder(os.Stdout).Encode(r)
		return
	}
	fmt.Printf("IPC sequential: mean %.2fus p50 %.2f p95 %.2f p99 %.2f\n", seq.MeanUS, seq.P50US, seq.P95US, seq.P99US)
	fmt.Printf("IPC concurrent: mean %.2fus p50 %.2f p95 %.2f p99 %.2f\n", con.MeanUS, con.P50US, con.P95US, con.P99US)
}

func runSeq(sock, token, session string, n int) latResult {
	d := make([]float64, 0, n)
	for i := 0; i < n; i++ {
		start := time.Now()
		one(sock, token, session)
		d = append(d, float64(time.Since(start).Microseconds()))
	}
	return summarize("ipc_seq", d)
}

func runCon(sock, token, session string, n, conc int) latResult {
	d := make([]float64, n)
	sem := make(chan struct{}, conc)
	var wg sync.WaitGroup
	for i := 0; i < n; i++ {
		i := i
		wg.Add(1)
		sem <- struct{}{}
		go func() {
			defer wg.Done()
			start := time.Now()
			one(sock, token, session)
			d[i] = float64(time.Since(start).Microseconds())
			<-sem
		}()
	}
	wg.Wait()
	return summarize("ipc_con", d)
}

func one(sock, token, session string) {
	c, err := net.Dial("unix", sock)
	if err != nil {
		panic(err)
	}
	defer c.Close()
	req := ipc.Request{V: 1, Op: "reserve", SessionID: session, PID: os.Getpid(), Bytes: 0, Token: token}
	if err := ipc.EncodeFrame(c, req); err != nil {
		panic(err)
	}
	var resp ipc.Response
	if err := ipc.DecodeFrame(c, &resp); err != nil || !resp.OK {
		panic("bad ipc response")
	}
}

func summarize(name string, values []float64) latResult {
	sort.Float64s(values)
	var sum float64
	for _, v := range values {
		sum += v
	}
	idx := func(p float64) int { return int(float64(len(values)-1) * p) }
	return latResult{Name: name, N: len(values), MeanUS: sum / float64(len(values)), P50US: values[idx(0.50)], P95US: values[idx(0.95)], P99US: values[idx(0.99)]}
}
