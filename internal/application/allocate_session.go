package application

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"gslice/internal/domain"
	"time"
)

func (s *Service) AllocateSession(ctx context.Context, limitBytes uint64, ttl time.Duration) (domain.Session, error) {
	b := make([]byte, 8)
	if _, err := rand.Read(b); err != nil {
		return domain.Session{}, err
	}
	session, err := domain.NewSession(hex.EncodeToString(b), limitBytes, s.clock.Now(), ttl)
	if err != nil {
		return domain.Session{}, err
	}
	if err := s.store.Create(ctx, session); err != nil {
		return domain.Session{}, err
	}
	s.metrics.SetSessionUsage(session.ID, 0)
	return session, nil
}
