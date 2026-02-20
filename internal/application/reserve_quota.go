package application

import (
	"context"
	"errors"
	"gslice/internal/domain"
)

type QuotaStatus struct {
	UsedBytes      uint64
	LimitBytes     uint64
	RemainingBytes uint64
}

func (s *Service) ReserveQuota(ctx context.Context, sessionID string, bytes uint64) (QuotaStatus, error) {
	session, err := s.store.Get(ctx, sessionID)
	if err != nil {
		return QuotaStatus{}, wrapDomainErr(err)
	}
	if session.IsExpired(s.clock.Now()) {
		return QuotaStatus{}, NewCodedError(CodeExpired, "session expired")
	}
	used, err := domain.Reserve(session.UsedBytes, session.VRAMLimitBytes, bytes)
	if err != nil {
		if errors.Is(err, domain.ErrQuotaExceeded) {
			return QuotaStatus{UsedBytes: session.UsedBytes, LimitBytes: session.VRAMLimitBytes, RemainingBytes: session.VRAMLimitBytes - session.UsedBytes}, NewCodedError(CodeQuotaExceeded, "quota exceeded")
		}
		return QuotaStatus{}, NewCodedError(CodeInternal, "internal error")
	}
	session.UsedBytes = used
	session.UpdatedAt = s.clock.Now()
	session.State = domain.SessionActive
	if err := s.store.UpdateSession(ctx, session); err != nil {
		return QuotaStatus{}, wrapDomainErr(err)
	}
	s.metrics.ObserveAllocationResult(true)
	s.metrics.SetSessionUsage(sessionID, float64(used))
	return QuotaStatus{UsedBytes: used, LimitBytes: session.VRAMLimitBytes, RemainingBytes: session.VRAMLimitBytes - used}, nil
}

func (s *Service) StatusQuota(ctx context.Context, sessionID string) (QuotaStatus, error) {
	session, err := s.store.Get(ctx, sessionID)
	if err != nil {
		return QuotaStatus{}, wrapDomainErr(err)
	}
	if session.IsExpired(s.clock.Now()) {
		return QuotaStatus{}, NewCodedError(CodeExpired, "session expired")
	}
	return QuotaStatus{UsedBytes: session.UsedBytes, LimitBytes: session.VRAMLimitBytes, RemainingBytes: session.VRAMLimitBytes - session.UsedBytes}, nil
}
