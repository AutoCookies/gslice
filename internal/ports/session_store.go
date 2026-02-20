package ports

import (
	"context"
	"gslice/internal/domain"
)

type SessionStore interface {
	Create(ctx context.Context, session domain.Session) error
	Get(ctx context.Context, id string) (domain.Session, error)
	List(ctx context.Context) ([]domain.Session, error)
	Delete(ctx context.Context, id string) error
	UpdateUsedBytes(ctx context.Context, id string, used uint64) error
}
