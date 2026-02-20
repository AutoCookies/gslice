package store

import (
	"context"
	"encoding/json"
	"fmt"
	"gslice/internal/domain"
	"gslice/internal/ports"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"
)

type stateFile struct {
	Sessions    map[string]domain.Session         `json:"sessions"`
	Allocations map[string]ports.AllocationRecord `json:"allocations"`
}

type BoltStore struct {
	path  string
	mu    sync.Mutex
	state stateFile
}

func NewBoltStore(path string) (*BoltStore, error) {
	s := &BoltStore{path: path, state: stateFile{Sessions: map[string]domain.Session{}, Allocations: map[string]ports.AllocationRecord{}}}
	b, err := os.ReadFile(path)
	if err == nil && len(b) > 0 {
		if err := json.Unmarshal(b, &s.state); err != nil {
			return nil, err
		}
		if s.state.Sessions == nil {
			s.state.Sessions = map[string]domain.Session{}
		}
		if s.state.Allocations == nil {
			s.state.Allocations = map[string]ports.AllocationRecord{}
		}
	}
	return s, nil
}

func allocKey(sessionID string, pid int, ptrID string) string {
	return sessionID + "|" + strconv.Itoa(pid) + "|" + ptrID
}

func (s *BoltStore) persist() error {
	b, err := json.MarshalIndent(s.state, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(s.path, b, 0o600)
}
func (s *BoltStore) Close() error { return nil }
func (s *BoltStore) Create(_ context.Context, session domain.Session) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.state.Sessions[session.ID] = session
	return s.persist()
}
func (s *BoltStore) Get(_ context.Context, id string) (domain.Session, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	v, ok := s.state.Sessions[id]
	if !ok {
		return domain.Session{}, domain.ErrSessionNotFound
	}
	return v, nil
}
func (s *BoltStore) List(_ context.Context) ([]domain.Session, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make([]domain.Session, 0, len(s.state.Sessions))
	for _, v := range s.state.Sessions {
		out = append(out, v)
	}
	return out, nil
}
func (s *BoltStore) Delete(_ context.Context, id string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	delete(s.state.Sessions, id)
	for k, r := range s.state.Allocations {
		if r.SessionID == id {
			delete(s.state.Allocations, k)
		}
	}
	return s.persist()
}
func (s *BoltStore) UpdateUsedBytes(_ context.Context, id string, used uint64) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	v, ok := s.state.Sessions[id]
	if !ok {
		return domain.ErrSessionNotFound
	}
	v.UsedBytes = used
	v.UpdatedAt = time.Now().UTC()
	s.state.Sessions[id] = v
	return s.persist()
}
func (s *BoltStore) UpdateSession(_ context.Context, sess domain.Session) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.state.Sessions[sess.ID]; !ok {
		return domain.ErrSessionNotFound
	}
	s.state.Sessions[sess.ID] = sess
	return s.persist()
}

func (s *BoltStore) RegisterAllocation(_ context.Context, rec ports.AllocationRecord) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	k := allocKey(rec.SessionID, rec.PID, rec.PtrID)
	rec.LastSeen = time.Now().UTC()
	s.state.Allocations[k] = rec
	return s.persist()
}

func (s *BoltStore) UnregisterAllocation(_ context.Context, sessionID string, pid int, ptrID string) (uint64, bool, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	k := allocKey(sessionID, pid, ptrID)
	r, ok := s.state.Allocations[k]
	if ok {
		delete(s.state.Allocations, k)
		if err := s.persist(); err != nil {
			return 0, false, err
		}
		return r.Bytes, true, nil
	}
	if err := s.persist(); err != nil {
		return 0, false, err
	}
	return 0, false, nil
}

func (s *BoltStore) ListAllocations(_ context.Context) ([]ports.AllocationRecord, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make([]ports.AllocationRecord, 0, len(s.state.Allocations))
	for _, v := range s.state.Allocations {
		out = append(out, v)
	}
	return out, nil
}

func (s *BoltStore) DeleteAllocationsForPID(_ context.Context, sessionID string, pid int) (uint64, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	var reclaimed uint64
	prefix := fmt.Sprintf("%s|%d|", sessionID, pid)
	for k, v := range s.state.Allocations {
		if strings.HasPrefix(k, prefix) {
			reclaimed += v.Bytes
			delete(s.state.Allocations, k)
		}
	}
	return reclaimed, s.persist()
}
