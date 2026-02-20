package tests

import (
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func TestFakeCUDAExampleApp(t *testing.T) {
	root := filepath.Join("..", "..")
	cmd := exec.Command("make", "build-native")
	cmd.Dir = root
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("build-native failed: %v\n%s", err, string(out))
	}

	run := exec.Command(filepath.Join(root, "native", "examples", "target_app"))
	run.Env = append(run.Environ(),
		"FAKECUDA_TOTAL_MEM=134217728",
		"ALLOC_BYTES=67108864",
		"ITERATIONS=5",
	)
	out, err := run.CombinedOutput()
	if err != nil {
		t.Fatalf("target_app failed: %v\n%s", err, string(out))
	}
	text := string(out)
	if strings.Count(text, "ALLOC_OK") != 2 {
		t.Fatalf("expected exactly 2 ALLOC_OK lines, got output:\n%s", text)
	}
	if !strings.Contains(text, "ALLOC_FAIL at iteration 2 code=2") {
		t.Fatalf("expected deterministic fail line, got output:\n%s", text)
	}
}
