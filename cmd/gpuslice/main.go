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
	"os/signal"
	"strconv"
	"syscall"
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
	client := apiClient{base: "http://127.0.0.1" + cfg.HTTPAddr}
	if len(os.Args) < 2 {
		usage()
		os.Exit(2)
	}
	switch os.Args[1] {
	case "allocate":
		limit, ttl := cfg.DefaultVRAM, int64(cfg.DefaultTTL.Seconds())
		if len(os.Args) > 2 {
			if v, e := strconv.ParseUint(os.Args[2], 10, 64); e == nil {
				limit = v
			}
		}
		if len(os.Args) > 3 {
			if v, e := strconv.ParseInt(os.Args[3], 10, 64); e == nil {
				ttl = v
			}
		}
		out, err := client.do(http.MethodPost, "/sessions", map[string]any{"limit_bytes": limit, "ttl_seconds": ttl})
		check(err, 1)
		fmt.Println(string(out))
	case "status":
		if len(os.Args) < 3 {
			usage()
			os.Exit(2)
		}
		out, err := client.do(http.MethodGet, "/sessions/"+os.Args[2], nil)
		check(err, 1)
		fmt.Println(string(out))
	case "release":
		if len(os.Args) < 3 {
			usage()
			os.Exit(2)
		}
		_, err := client.do(http.MethodDelete, "/sessions/"+os.Args[2], nil)
		check(err, 1)
	case "list":
		out, err := client.do(http.MethodGet, "/sessions", nil)
		check(err, 1)
		fmt.Println(string(out))
	case "run":
		if len(os.Args) < 7 {
			fmt.Fprintln(os.Stderr, "run <limit_bytes> <ttl_sec> <preload> <lib_path> <command...>")
			os.Exit(2)
		}
		limit, _ := strconv.ParseUint(os.Args[2], 10, 64)
		ttl, _ := strconv.ParseInt(os.Args[3], 10, 64)
		allocOut, err := client.do(http.MethodPost, "/sessions", map[string]any{"limit_bytes": limit, "ttl_seconds": ttl})
		check(err, 1)
		var sess map[string]any
		_ = json.Unmarshal(allocOut, &sess)
		id, _ := sess["session_id"].(string)
		if id == "" {
			fmt.Fprintln(os.Stderr, "failed to parse session id")
			os.Exit(1)
		}
		defer func() { _, _ = client.do(http.MethodDelete, "/sessions/"+id, nil) }()
		st, _ := store.NewBoltStore(cfg.DBPath)
		defer st.Close()
		svc := application.NewService(st, application.RealClock{}, metrics.New(false), logging.New())
		ctx, cancel := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
		defer cancel()
		err = svc.RunCommand(ctx, id, cfg.IPCSocketPath, cfg.IPCToken, os.Args[4], os.Args[5], os.Args[6:])
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
	case "run-existing":
		if len(os.Args) < 6 {
			fmt.Fprintln(os.Stderr, "run-existing <session_id> <preload> <lib_path> <command...>")
			os.Exit(2)
		}
		st, _ := store.NewBoltStore(cfg.DBPath)
		defer st.Close()
		svc := application.NewService(st, application.RealClock{}, metrics.New(false), logging.New())
		ctx, cancel := context.WithTimeout(context.Background(), 24*time.Hour)
		defer cancel()
		check(svc.RunCommand(ctx, os.Args[2], cfg.IPCSocketPath, cfg.IPCToken, os.Args[3], os.Args[4], os.Args[5:]), 1)
	default:
		usage()
		os.Exit(2)
	}
}
func usage() {
	fmt.Println("usage: gpuslice [allocate [limit ttl]|status <id>|release <id>|list|run <limit> <ttl> <preload> <lib_path> <cmd...>|run-existing <session> <preload> <lib_path> <cmd...>]")
}
func check(err error, code int) {
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(code)
	}
}
