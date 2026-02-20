package application

import (
	"context"
	"gslice/internal/domain"
)

func (s *Service) GetStatus(ctx context.Context, id string) (domain.Session, error) {
	return s.store.Get(ctx, id)
}

func (s *Service) ListSessions(ctx context.Context) ([]domain.Session, error) {
	return s.store.List(ctx)
}
