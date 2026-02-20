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
	used := domain.Release(session.UsedBytes, bytes)
	if err := s.store.UpdateUsedBytes(ctx, sessionID, used); err != nil {
		return QuotaStatus{}, wrapDomainErr(err)
	}
	return QuotaStatus{UsedBytes: used, LimitBytes: session.VRAMLimitBytes, RemainingBytes: session.VRAMLimitBytes - used}, nil
}
