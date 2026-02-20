package config

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"time"
)

type Config struct {
	HTTPAddr         string
	IPCSocketPath    string
	DBPath           string
	DefaultTTL       time.Duration
	DefaultVRAM      uint64
	IPCToken         string
	RequireIPCToken  bool
	RecoveryInterval time.Duration
	AuditPath        string
	MetricsDebug     bool
	IPCRatePerSecond int
}

func Load() (Config, error) {
	cfg := Config{
		HTTPAddr:         envOrDefault("GPUSLICE_HTTP_ADDR", ":8080"),
		IPCSocketPath:    envOrDefault("GPUSLICE_IPC_SOCK", "/tmp/gpusliced.sock"),
		DBPath:           envOrDefault("GPUSLICE_DB_PATH", "./gpuslice.db"),
		DefaultTTL:       envDurationOrDefault("GPUSLICE_DEFAULT_TTL", 15*time.Minute),
		DefaultVRAM:      envUintOrDefault("GPUSLICE_DEFAULT_VRAM", 64*1024*1024),
		IPCToken:         os.Getenv("GPUSLICE_IPC_TOKEN"),
		RequireIPCToken:  envBool("GPUSLICE_IPC_TOKEN_REQUIRED", false),
		RecoveryInterval: envDurationOrDefault("GPUSLICE_RECOVERY_INTERVAL", 2*time.Second),
		AuditPath:        os.Getenv("GPUSLICE_AUDIT_PATH"),
		MetricsDebug:     envBool("GPUSLICE_METRICS_DEBUG", false),
		IPCRatePerSecond: envInt("GPUSLICE_IPC_RATE_PER_SEC", 1000),
	}
	if err := os.MkdirAll(filepath.Dir(cfg.DBPath), 0o755); err != nil && filepath.Dir(cfg.DBPath) != "." {
		return Config{}, fmt.Errorf("create db dir: %w", err)
	}
	if len(cfg.IPCSocketPath) >= 104 {
		return Config{}, fmt.Errorf("socket path too long")
	}
	if cfg.RequireIPCToken && cfg.IPCToken == "" {
		return Config{}, fmt.Errorf("ipc token required but empty")
	}
	if cfg.IPCRatePerSecond <= 0 {
		cfg.IPCRatePerSecond = 1000
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
		if p, e := time.ParseDuration(v); e == nil {
			return p
		}
	}
	return d
}
func envUintOrDefault(k string, d uint64) uint64 {
	if v := os.Getenv(k); v != "" {
		if p, e := strconv.ParseUint(v, 10, 64); e == nil {
			return p
		}
	}
	return d
}
func envBool(k string, d bool) bool {
	if v := os.Getenv(k); v != "" {
		return v == "1" || v == "true"
	}
	return d
}
func envInt(k string, d int) int {
	if v := os.Getenv(k); v != "" {
		if p, e := strconv.Atoi(v); e == nil {
			return p
		}
	}
	return d
}
