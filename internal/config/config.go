package config

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"time"
)

type Config struct {
	HTTPAddr      string
	IPCSocketPath string
	DBPath        string
	DefaultTTL    time.Duration
	DefaultVRAM   uint64
}

func Load() (Config, error) {
	cfg := Config{
		HTTPAddr:      envOrDefault("GPUSLICE_HTTP_ADDR", ":8080"),
		IPCSocketPath: envOrDefault("GPUSLICE_IPC_SOCK", "/tmp/gpusliced.sock"),
		DBPath:        envOrDefault("GPUSLICE_DB_PATH", "./gpuslice.db"),
		DefaultTTL:    envDurationOrDefault("GPUSLICE_DEFAULT_TTL", 10*time.Minute),
		DefaultVRAM:   envUintOrDefault("GPUSLICE_DEFAULT_VRAM", 64*1024*1024),
	}
	if err := os.MkdirAll(filepath.Dir(cfg.DBPath), 0o755); err != nil && filepath.Dir(cfg.DBPath) != "." {
		return Config{}, fmt.Errorf("create db dir: %w", err)
	}
	if len(cfg.IPCSocketPath) >= 104 {
		return Config{}, fmt.Errorf("socket path too long")
	}
	return cfg, nil
}

func envOrDefault(k, d string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return d
}

func envDurationOrDefault(k string, d time.Duration) time.Duration {
	if v := os.Getenv(k); v != "" {
		if parsed, err := time.ParseDuration(v); err == nil {
			return parsed
		}
	}
	return d
}

func envUintOrDefault(k string, d uint64) uint64 {
	if v := os.Getenv(k); v != "" {
		if parsed, err := strconv.ParseUint(v, 10, 64); err == nil {
			return parsed
		}
	}
	return d
}
