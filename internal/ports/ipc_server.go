package ports

import "context"

type IPCServer interface {
	Start(ctx context.Context) error
	Stop() error
}
