package metrics

import (
	"fmt"
	"sort"
	"strings"
	"sync"
)

type PromMetrics struct {
	mu           sync.Mutex
	sessionUsage map[string]float64
	allowedCount uint64
	deniedCount  uint64
}

func New() *PromMetrics {
	return &PromMetrics{sessionUsage: map[string]float64{}}
}

func (m *PromMetrics) SetSessionUsage(sessionID string, used float64) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.sessionUsage[sessionID] = used
}
func (m *PromMetrics) ObserveAllocationResult(allowed bool) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if allowed {
		m.allowedCount++
	} else {
		m.deniedCount++
	}
}

func (m *PromMetrics) RenderPrometheus() string {
	m.mu.Lock()
	defer m.mu.Unlock()
	var b strings.Builder
	b.WriteString("# TYPE gpuslice_allocations_total counter\n")
	b.WriteString(fmt.Sprintf("gpuslice_allocations_total{allowed=\"true\"} %d\n", m.allowedCount))
	b.WriteString(fmt.Sprintf("gpuslice_allocations_total{allowed=\"false\"} %d\n", m.deniedCount))
	b.WriteString("# TYPE gpuslice_session_used_bytes gauge\n")
	keys := make([]string, 0, len(m.sessionUsage))
	for k := range m.sessionUsage {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		b.WriteString(fmt.Sprintf("gpuslice_session_used_bytes{session_id=\"%s\"} %.0f\n", k, m.sessionUsage[k]))
	}
	return b.String()
}
