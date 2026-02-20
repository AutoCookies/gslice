package ipc

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
)

const (
	ProtocolVersion = 1
	maxMessageSize  = 64 * 1024
)

type Request struct {
	V         int    `json:"v"`
	Op        string `json:"op"`
	SessionID string `json:"session_id"`
	PID       int    `json:"pid"`
	Bytes     int64  `json:"bytes"`
}

type Response struct {
	V              int           `json:"v"`
	OK             bool          `json:"ok"`
	Error          *ErrorPayload `json:"error"`
	UsedBytes      uint64        `json:"used_bytes"`
	LimitBytes     uint64        `json:"limit_bytes"`
	RemainingBytes uint64        `json:"remaining_bytes"`
}

func ValidateRequest(req Request) *ErrorPayload {
	if req.V != ProtocolVersion {
		return &ErrorPayload{Code: CodeUnsupportedVersion, Message: "unsupported protocol version"}
	}
	if req.Op != "reserve" && req.Op != "release" && req.Op != "status" {
		return &ErrorPayload{Code: CodeBadOp, Message: "unsupported operation"}
	}
	if req.SessionID == "" {
		return &ErrorPayload{Code: CodeBadRequest, Message: "session_id is required"}
	}
	if req.PID <= 0 {
		return &ErrorPayload{Code: CodeBadRequest, Message: "pid must be > 0"}
	}
	if req.Bytes < 0 {
		return &ErrorPayload{Code: CodeBadRequest, Message: "bytes must be >= 0"}
	}
	return nil
}

func EncodeFrame(w io.Writer, v any) error {
	body, err := json.Marshal(v)
	if err != nil {
		return fmt.Errorf("marshal: %w", err)
	}
	if len(body) > maxMessageSize {
		return fmt.Errorf("frame too large")
	}
	var hdr [4]byte
	binary.BigEndian.PutUint32(hdr[:], uint32(len(body)))
	if _, err := w.Write(hdr[:]); err != nil {
		return err
	}
	_, err = w.Write(body)
	return err
}

func DecodeFrame(r io.Reader, out any) error {
	var hdr [4]byte
	if _, err := io.ReadFull(r, hdr[:]); err != nil {
		return err
	}
	n := binary.BigEndian.Uint32(hdr[:])
	if n == 0 || n > maxMessageSize {
		return fmt.Errorf("invalid frame size")
	}
	buf := make([]byte, n)
	if _, err := io.ReadFull(r, buf); err != nil {
		return err
	}
	if err := json.Unmarshal(buf, out); err != nil {
		return fmt.Errorf("unmarshal: %w", err)
	}
	return nil
}
