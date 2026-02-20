package ipc

import (
	"context"
	"crypto/subtle"
	"errors"
	"gslice/internal/application"
	"gslice/internal/ports"
	"io"
	"net"
	"os"
	"path/filepath"
	"sync"
	"time"
)

type UDSServer struct {
	path         string
	service      *application.Service
	logger       ports.Logger
	ln           net.Listener
	wg           sync.WaitGroup
	token        string
	requireToken bool
	ratePerSec   int
}

func NewUDSServer(path string, service *application.Service, logger ports.Logger, token string, requireToken bool, ratePerSec int) *UDSServer {
	if ratePerSec <= 0 {
		ratePerSec = 1000
	}
	return &UDSServer{path: path, service: service, logger: logger, token: token, requireToken: requireToken, ratePerSec: ratePerSec}
}

func (u *UDSServer) Start(ctx context.Context) error {
	if err := os.MkdirAll(filepath.Dir(u.path), 0o755); err != nil && filepath.Dir(u.path) != "." {
		return err
	}
	if err := os.Remove(u.path); err != nil && !errors.Is(err, os.ErrNotExist) {
		return err
	}
	ln, err := net.Listen("unix", u.path)
	if err != nil {
		return err
	}
	if err := os.Chmod(u.path, 0o600); err != nil {
		_ = ln.Close()
		return err
	}
	u.ln = ln
	u.wg.Add(1)
	go u.acceptLoop(ctx)
	return nil
}

func (u *UDSServer) acceptLoop(ctx context.Context) {
	defer u.wg.Done()
	for {
		conn, err := u.ln.Accept()
		if err != nil {
			if errors.Is(err, net.ErrClosed) {
				return
			}
			select {
			case <-ctx.Done():
				return
			default:
			}
			if u.logger != nil {
				u.logger.Error("ipc_accept_failed", "error", err.Error())
			}
			return
		}
		u.wg.Add(1)
		go u.handleConn(ctx, conn)
	}
}

func (u *UDSServer) Stop() error {
	if u.ln != nil {
		_ = u.ln.Close()
	}
	u.wg.Wait()
	if err := os.Remove(u.path); err != nil && !errors.Is(err, os.ErrNotExist) {
		return err
	}
	return nil
}

func (u *UDSServer) handleConn(ctx context.Context, conn net.Conn) {
	defer u.wg.Done()
	defer conn.Close()
	windowStart := time.Now()
	reqCount := 0
	for {
		if time.Since(windowStart) >= time.Second {
			windowStart = time.Now()
			reqCount = 0
		}
		reqCount++
		if reqCount > u.ratePerSec {
			return
		}
		select {
		case <-ctx.Done():
			return
		default:
		}
		var req Request
		if err := DecodeFrame(conn, &req); err != nil {
			if errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF) {
				return
			}
			_ = EncodeFrame(conn, Response{V: ProtocolVersion, OK: false, Error: &ErrorPayload{Code: CodeBadRequest, Message: "malformed frame"}})
			return
		}
		resp := u.dispatch(ctx, req)
		if err := EncodeFrame(conn, resp); err != nil {
			return
		}
	}
}

func (u *UDSServer) authorized(token string) bool {
	if !u.requireToken && u.token == "" {
		return true
	}
	if u.token == "" {
		return false
	}
	if len(token) != len(u.token) {
		return false
	}
	return subtle.ConstantTimeCompare([]byte(token), []byte(u.token)) == 1
}

func (u *UDSServer) dispatch(ctx context.Context, req Request) Response {
	if verr := ValidateRequest(req); verr != nil {
		return Response{V: ProtocolVersion, OK: false, Error: verr}
	}
	if !u.authorized(req.Token) {
		return Response{V: ProtocolVersion, OK: false, Error: &ErrorPayload{Code: CodeUnauthorized, Message: "unauthorized"}}
	}
	bytes := uint64(req.Bytes)
	switch req.Op {
	case "reserve":
		st, err := u.service.ReserveQuota(ctx, req.SessionID, bytes)
		return fromAppResult(st, err)
	case "release":
		st, err := u.service.ReleaseQuota(ctx, req.SessionID, bytes)
		return fromAppResult(st, err)
	case "status":
		st, err := u.service.StatusQuota(ctx, req.SessionID)
		return fromAppResult(st, err)
	case "alloc_register":
		err := u.service.RegisterAllocation(ctx, req.SessionID, req.PID, req.PtrID, bytes)
		if err != nil {
			return fromAppResult(application.QuotaStatus{}, err)
		}
		st, _ := u.service.StatusQuota(ctx, req.SessionID)
		return fromAppResult(st, nil)
	case "alloc_unregister":
		err := u.service.UnregisterAllocation(ctx, req.SessionID, req.PID, req.PtrID)
		if err != nil {
			return fromAppResult(application.QuotaStatus{}, err)
		}
		st, _ := u.service.StatusQuota(ctx, req.SessionID)
		return fromAppResult(st, nil)
	default:
		return Response{V: ProtocolVersion, OK: false, Error: &ErrorPayload{Code: CodeBadOp, Message: "unsupported operation"}}
	}
}

func fromAppResult(status application.QuotaStatus, err error) Response {
	if err != nil {
		return Response{V: ProtocolVersion, OK: false, Error: &ErrorPayload{Code: application.ErrorCode(err), Message: err.Error()}, UsedBytes: status.UsedBytes, LimitBytes: status.LimitBytes, RemainingBytes: status.RemainingBytes}
	}
	return Response{V: ProtocolVersion, OK: true, UsedBytes: status.UsedBytes, LimitBytes: status.LimitBytes, RemainingBytes: status.RemainingBytes}
}
