package metrics

import (
	"fmt"
	"sort"
	"strings"
	"sync"
)

type PromMetrics struct {
	mu                  sync.Mutex
	debugLabels         bool
	sessionUsage        map[string]float64
	allowedCount        uint64
	deniedCount         uint64
	recoveredBytesTotal uint64
	allocEventsTotal    uint64
}

func New(debug bool) *PromMetrics {
	return &PromMetrics{debugLabels: debug, sessionUsage: map[string]float64{}}
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
func (m *PromMetrics) ObserveAllocEvent() { m.mu.Lock(); defer m.mu.Unlock(); m.allocEventsTotal++ }
func (m *PromMetrics) ObserveRecoveredBytes(v uint64) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.recoveredBytesTotal += v
}

func (m *PromMetrics) RenderPrometheus() string {
	m.mu.Lock()
	defer m.mu.Unlock()
	var b strings.Builder
	var total float64
	for _, v := range m.sessionUsage {
		total += v
	}
	active := 0
	for _, v := range m.sessionUsage {
		if v >= 0 {
			active++
		}
	}
	b.WriteString("# TYPE gpuslice_sessions_active gauge\n")
	b.WriteString(fmt.Sprintf("gpuslice_sessions_active %d\n", active))
	b.WriteString("# TYPE gpuslice_used_bytes_total gauge\n")
	b.WriteString(fmt.Sprintf("gpuslice_used_bytes_total %.0f\n", total))
	b.WriteString("# TYPE gpuslice_alloc_events_total counter\n")
	b.WriteString(fmt.Sprintf("gpuslice_alloc_events_total %d\n", m.allocEventsTotal))
	b.WriteString("# TYPE gpuslice_denied_alloc_total counter\n")
	b.WriteString(fmt.Sprintf("gpuslice_denied_alloc_total %d\n", m.deniedCount))
	b.WriteString("# TYPE gpuslice_recovered_bytes_total counter\n")
	b.WriteString(fmt.Sprintf("gpuslice_recovered_bytes_total %d\n", m.recoveredBytesTotal))
	if m.debugLabels {
		b.WriteString("# TYPE gpuslice_used_bytes gauge\n")
		keys := make([]string, 0, len(m.sessionUsage))
		for k := range m.sessionUsage {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		for _, k := range keys {
			b.WriteString(fmt.Sprintf("gpuslice_used_bytes{session_id=\"%s\"} %.0f\n", k, m.sessionUsage[k]))
		}
	}
	return b.String()
}
