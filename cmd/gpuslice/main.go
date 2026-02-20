package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"gslice/internal/adapters/logging"
	"gslice/internal/adapters/metrics"
	"gslice/internal/adapters/store"
	"gslice/internal/application"
	"gslice/internal/config"
	"io"
	"net/http"
	"os"
	"strconv"
	"time"
)

type apiClient struct{ base string }

func (c apiClient) do(method, path string, body any) ([]byte, error) {
	var rd io.Reader
	if body != nil {
		b, _ := json.Marshal(body)
		rd = bytes.NewReader(b)
	}
	req, _ := http.NewRequest(method, c.base+path, rd)
	req.Header.Set("Content-Type", "application/json")
	res, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	out, _ := io.ReadAll(res.Body)
	if res.StatusCode >= 300 {
		return nil, fmt.Errorf("status %d: %s", res.StatusCode, string(out))
	}
	return out, nil
}

func main() {
	cfg, _ := config.Load()
	apiAddr := "http://127.0.0.1" + cfg.HTTPAddr
	if len(os.Args) < 2 {
		usage()
		os.Exit(1)
	}
	cmd := os.Args[1]
	client := apiClient{base: apiAddr}
	switch cmd {
	case "allocate":
		limit := cfg.DefaultVRAM
		ttl := int64(cfg.DefaultTTL.Seconds())
		if len(os.Args) > 2 {
			if v, err := strconv.ParseUint(os.Args[2], 10, 64); err == nil {
				limit = v
			}
		}
		if len(os.Args) > 3 {
			if v, err := strconv.ParseInt(os.Args[3], 10, 64); err == nil {
				ttl = v
			}
		}
		out, err := client.do(http.MethodPost, "/sessions", map[string]any{"limit_bytes": limit, "ttl_seconds": ttl})
		check(err)
		fmt.Println(string(out))
	case "status":
		if len(os.Args) < 3 {
			usage()
			os.Exit(1)
		}
		out, err := client.do(http.MethodGet, "/sessions/"+os.Args[2], nil)
		check(err)
		fmt.Println(string(out))
	case "release":
		if len(os.Args) < 3 {
			usage()
			os.Exit(1)
		}
		_, err := client.do(http.MethodDelete, "/sessions/"+os.Args[2], nil)
		check(err)
	case "list":
		out, err := client.do(http.MethodGet, "/sessions", nil)
		check(err)
		fmt.Println(string(out))
	case "run":
		if len(os.Args) < 6 {
			fmt.Fprintln(os.Stderr, "run <session_id> <preload> <lib_path> <command...>")
			os.Exit(1)
		}
		st, _ := store.NewBoltStore(cfg.DBPath)
		defer st.Close()
		svc := application.NewService(st, application.RealClock{}, metrics.New(), logging.New())
		ctx, cancel := context.WithTimeout(context.Background(), 24*time.Hour)
		defer cancel()
		check(svc.RunCommand(ctx, os.Args[2], cfg.IPCSocketPath, os.Args[3], os.Args[4], os.Args[5:]))
	default:
		usage()
		os.Exit(1)
	}
}
func usage() {
	fmt.Println("usage: gpuslice [allocate [limit ttl]|status <id>|release <id>|list|run <session> <preload> <lib_path> <cmd...>]")
}
func check(err error) {
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
