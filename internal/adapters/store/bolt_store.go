package store

import (
	"context"
	"encoding/json"
	"gslice/internal/domain"
	"os"
	"sync"
)

type BoltStore struct {
	path     string
	mu       sync.Mutex
	sessions map[string]domain.Session
}

func NewBoltStore(path string) (*BoltStore, error) {
	s := &BoltStore{path: path, sessions: map[string]domain.Session{}}
	b, err := os.ReadFile(path)
	if err == nil && len(b) > 0 {
		if err := json.Unmarshal(b, &s.sessions); err != nil {
			return nil, err
		}
	}
	return s, nil
}

func (s *BoltStore) persist() error {
	b, err := json.MarshalIndent(s.sessions, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(s.path, b, 0o600)
}
func (s *BoltStore) Close() error { return nil }
func (s *BoltStore) Create(_ context.Context, session domain.Session) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.sessions[session.ID] = session
	return s.persist()
}
func (s *BoltStore) Get(_ context.Context, id string) (domain.Session, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	v, ok := s.sessions[id]
	if !ok {
		return domain.Session{}, domain.ErrSessionNotFound
	}
	return v, nil
}
func (s *BoltStore) List(_ context.Context) ([]domain.Session, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make([]domain.Session, 0, len(s.sessions))
	for _, v := range s.sessions {
		out = append(out, v)
	}
	return out, nil
}
func (s *BoltStore) Delete(_ context.Context, id string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	delete(s.sessions, id)
	return s.persist()
}
func (s *BoltStore) UpdateUsedBytes(_ context.Context, id string, used uint64) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	v, ok := s.sessions[id]
	if !ok {
		return domain.ErrSessionNotFound
	}
	v.UsedBytes = used
	s.sessions[id] = v
	return s.persist()
}
