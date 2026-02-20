package application

import (
	"context"
	"gslice/internal/domain"
)

func (s *Service) ReleaseQuota(ctx context.Context, sessionID string, bytes uint64) (QuotaStatus, error) {
	session, err := s.store.Get(ctx, sessionID)
	if err != nil {
		return QuotaStatus{}, wrapDomainErr(err)
	}
	if session.IsExpired(s.clock.Now()) {
		return QuotaStatus{}, NewCodedError(CodeExpired, "session expired")
	}
	session.UsedBytes = domain.Release(session.UsedBytes, bytes)
	session.UpdatedAt = s.clock.Now()
	if err := s.store.UpdateSession(ctx, session); err != nil {
		return QuotaStatus{}, wrapDomainErr(err)
	}
	s.metrics.SetSessionUsage(sessionID, float64(session.UsedBytes))
	return QuotaStatus{UsedBytes: session.UsedBytes, LimitBytes: session.VRAMLimitBytes, RemainingBytes: session.VRAMLimitBytes - session.UsedBytes}, nil
}
