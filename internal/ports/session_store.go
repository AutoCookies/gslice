package ports

import (
	"context"
	"gslice/internal/domain"
	"time"
)

type AllocationRecord struct {
	SessionID string    `json:"session_id"`
	PID       int       `json:"pid"`
	PtrID     string    `json:"ptr_id"`
	Bytes     uint64    `json:"bytes"`
	LastSeen  time.Time `json:"last_seen"`
}

type SessionStore interface {
	Create(ctx context.Context, session domain.Session) error
	Get(ctx context.Context, id string) (domain.Session, error)
	List(ctx context.Context) ([]domain.Session, error)
	Delete(ctx context.Context, id string) error
	UpdateUsedBytes(ctx context.Context, id string, used uint64) error
	UpdateSession(ctx context.Context, session domain.Session) error

	RegisterAllocation(ctx context.Context, rec AllocationRecord) error
	UnregisterAllocation(ctx context.Context, sessionID string, pid int, ptrID string) (uint64, bool, error)
	ListAllocations(ctx context.Context) ([]AllocationRecord, error)
	DeleteAllocationsForPID(ctx context.Context, sessionID string, pid int) (uint64, error)
}
