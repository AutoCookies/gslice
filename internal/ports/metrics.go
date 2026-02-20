package ports

type Metrics interface {
	SetSessionUsage(sessionID string, used float64)
	ObserveAllocationResult(allowed bool)
}
