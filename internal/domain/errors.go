package domain

import "errors"

var (
	ErrInvalidSessionLimit = errors.New("invalid session limit")
	ErrInvalidSessionID    = errors.New("invalid session id")
	ErrSessionExpired      = errors.New("session expired")
	ErrSessionNotFound     = errors.New("session not found")
	ErrQuotaExceeded       = errors.New("quota exceeded")
	ErrUnknownPointer      = errors.New("unknown pointer")
)
