package application

import "context"

func (s *Service) ReleaseSession(ctx context.Context, id string) error {
	return s.store.Delete(ctx, id)
}
