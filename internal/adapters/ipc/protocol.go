package ipc

type Request struct {
	PID       int    `json:"pid"`
	SessionID string `json:"session_id"`
	Op        string `json:"op"`
	Bytes     uint64 `json:"bytes"`
}

type Response struct {
	Allowed        bool   `json:"allowed"`
	UsedBytes      uint64 `json:"used_bytes"`
	RemainingBytes uint64 `json:"remaining_bytes"`
	Error          string `json:"error,omitempty"`
}
