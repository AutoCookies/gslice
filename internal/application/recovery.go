package application

import (
	"context"
	"gslice/internal/domain"
	"gslice/internal/ports"
	"os"
	"path/filepath"
	"strconv"
	"time"
)

func (s *Service) RegisterAllocation(ctx context.Context, sessionID string, pid int, ptrID string, bytes uint64) error {
	sess, err := s.store.Get(ctx, sessionID)
	if err != nil {
		return wrapDomainErr(err)
	}
	if sess.IsExpired(s.clock.Now()) {
		return NewCodedError(CodeExpired, "session expired")
	}
	rec := ports.AllocationRecord{SessionID: sessionID, PID: pid, PtrID: ptrID, Bytes: bytes, LastSeen: s.clock.Now()}
	if obs, ok := s.metrics.(interface{ ObserveAllocEvent() }); ok {
		obs.ObserveAllocEvent()
	}
	return s.store.RegisterAllocation(ctx, rec)
}

func (s *Service) UnregisterAllocation(ctx context.Context, sessionID string, pid int, ptrID string) error {
	_, _, err := s.store.UnregisterAllocation(ctx, sessionID, pid, ptrID)
	return err
}

func (s *Service) ReapExpiredSessions(ctx context.Context) error {
	sessions, err := s.store.List(ctx)
	if err != nil {
		return err
	}
	now := s.clock.Now()
	for _, sess := range sessions {
		if sess.State == domain.SessionActive && now.After(sess.ExpiresAt) {
			sess.State = domain.SessionExpired
			sess.UsedBytes = 0
			sess.UpdatedAt = now
			if err := s.store.UpdateSession(ctx, sess); err != nil {
				return err
			}
		}
	}
	return nil
}

func (s *Service) RecoverOrphanedAllocations(ctx context.Context) (uint64, error) {
	recs, err := s.store.ListAllocations(ctx)
	if err != nil {
		return 0, err
	}
	type k struct {
		sid string
		pid int
	}
	seen := map[k]bool{}
	var recovered uint64
	for _, rec := range recs {
		key := k{rec.SessionID, rec.PID}
		if seen[key] {
			continue
		}
		seen[key] = true
		if isPIDAlive(rec.PID) {
			continue
		}
		reclaimed, err := s.store.DeleteAllocationsForPID(ctx, rec.SessionID, rec.PID)
		if err != nil {
			return recovered, err
		}
		sess, err := s.store.Get(ctx, rec.SessionID)
		if err == nil {
			sess.UsedBytes = domain.Release(sess.UsedBytes, reclaimed)
			sess.UpdatedAt = s.clock.Now()
			_ = s.store.UpdateSession(ctx, sess)
		}
		if obs, ok := s.metrics.(interface{ ObserveRecoveredBytes(uint64) }); ok {
			obs.ObserveRecoveredBytes(reclaimed)
		}
		recovered += reclaimed
	}
	return recovered, nil
}

func isPIDAlive(pid int) bool {
	if pid <= 0 {
		return false
	}
	_, err := os.Stat(filepath.Join("/proc", strconv.Itoa(pid)))
	return err == nil
}

func (s *Service) StartBackground(ctx context.Context, interval time.Duration) {
	t := time.NewTicker(interval)
	go func() {
		defer t.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-t.C:
				_ = s.ReapExpiredSessions(ctx)
				_, _ = s.RecoverOrphanedAllocations(ctx)
			}
		}
	}()
}
