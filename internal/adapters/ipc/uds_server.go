package ipc

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"gslice/internal/application"
	"net"
	"os"
	"strings"
	"sync"
)

type UDSServer struct {
	path    string
	service *application.Service
	ln      net.Listener
	wg      sync.WaitGroup
}

func NewUDSServer(path string, service *application.Service) *UDSServer {
	return &UDSServer{path: path, service: service}
}

func (u *UDSServer) Start(ctx context.Context) error {
	_ = os.Remove(u.path)
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
	go func() {
		defer u.wg.Done()
		for {
			conn, err := ln.Accept()
			if err != nil {
				if errors.Is(err, net.ErrClosed) || strings.Contains(err.Error(), "closed") {
					return
				}
				continue
			}
			u.wg.Add(1)
			go func(c net.Conn) {
				defer u.wg.Done()
				defer c.Close()
				u.handle(ctx, c)
			}(conn)
		}
	}()
	return nil
}

func (u *UDSServer) Stop() error {
	if u.ln != nil {
		_ = u.ln.Close()
	}
	u.wg.Wait()
	_ = os.Remove(u.path)
	return nil
}

func (u *UDSServer) handle(ctx context.Context, conn net.Conn) {
	reader := bufio.NewReader(conn)
	line, err := reader.ReadBytes('\n')
	if err != nil {
		return
	}
	var req Request
	if err := json.Unmarshal(line, &req); err != nil {
		_ = json.NewEncoder(conn).Encode(Response{Allowed: false, Error: "bad_request"})
		return
	}
	if req.SessionID == "" || req.Bytes == 0 {
		_ = json.NewEncoder(conn).Encode(Response{Allowed: false, Error: "invalid_input"})
		return
	}
	switch req.Op {
	case "reserve":
		res, err := u.service.Reserve(ctx, req.SessionID, req.Bytes)
		if err != nil {
			_ = json.NewEncoder(conn).Encode(Response{Allowed: false, Error: err.Error()})
			return
		}
		_ = json.NewEncoder(conn).Encode(Response{Allowed: res.Allowed, UsedBytes: res.UsedBytes, RemainingBytes: res.RemainingBytes})
	case "release":
		res, err := u.service.ReleaseBytes(ctx, req.SessionID, req.Bytes)
		if err != nil {
			_ = json.NewEncoder(conn).Encode(Response{Allowed: false, Error: err.Error()})
			return
		}
		_ = json.NewEncoder(conn).Encode(Response{Allowed: res.Allowed, UsedBytes: res.UsedBytes, RemainingBytes: res.RemainingBytes})
	default:
		_ = json.NewEncoder(conn).Encode(Response{Allowed: false, Error: "unsupported_op"})
	}
}
