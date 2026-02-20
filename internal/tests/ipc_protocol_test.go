package tests

import (
	"bytes"
	"gslice/internal/adapters/ipc"
	"testing"
)

func TestProtocolRoundTrip(t *testing.T) {
	buf := bytes.NewBuffer(nil)
	in := ipc.Request{V: 1, Op: "reserve", SessionID: "s1", PID: 123, Bytes: 1024}
	if err := ipc.EncodeFrame(buf, in); err != nil {
		t.Fatal(err)
	}
	var out ipc.Request
	if err := ipc.DecodeFrame(buf, &out); err != nil {
		t.Fatal(err)
	}
	if out != in {
		t.Fatalf("round trip mismatch: %#v %#v", in, out)
	}
}

func TestProtocolValidation(t *testing.T) {
	cases := []struct {
		name string
		req  ipc.Request
		code string
	}{
		{"invalid_version", ipc.Request{V: 2, Op: "status", SessionID: "s", PID: 1, Bytes: 0}, ipc.CodeUnsupportedVersion},
		{"unknown_op", ipc.Request{V: 1, Op: "oops", SessionID: "s", PID: 1, Bytes: 0}, ipc.CodeBadOp},
		{"missing_session", ipc.Request{V: 1, Op: "status", SessionID: "", PID: 1, Bytes: 0}, ipc.CodeBadRequest},
		{"negative_bytes", ipc.Request{V: 1, Op: "reserve", SessionID: "s", PID: 1, Bytes: -1}, ipc.CodeBadRequest},
		{"zero_pid", ipc.Request{V: 1, Op: "reserve", SessionID: "s", PID: 0, Bytes: 0}, ipc.CodeBadRequest},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := ipc.ValidateRequest(tc.req)
			if err == nil || err.Code != tc.code {
				t.Fatalf("expected %s got %#v", tc.code, err)
			}
		})
	}
}
