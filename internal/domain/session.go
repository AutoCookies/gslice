package domain

import "time"

type SessionState string

const (
	SessionActive  SessionState = "ACTIVE"
	SessionExpired SessionState = "EXPIRED"
	SessionRevoked SessionState = "REVOKED"
)

type Session struct {
	ID             string       `json:"session_id"`
	VRAMLimitBytes uint64       `json:"vram_limit_bytes"`
	UsedBytes      uint64       `json:"used_bytes"`
	CreatedAt      time.Time    `json:"created_at"`
	UpdatedAt      time.Time    `json:"updated_at"`
	ExpiresAt      time.Time    `json:"expires_at"`
	State          SessionState `json:"state"`
}

func NewSession(id string, limitBytes uint64, now time.Time, ttl time.Duration) (Session, error) {
	if id == "" {
		return Session{}, ErrInvalidSessionID
	}
	if limitBytes == 0 {
		return Session{}, ErrInvalidSessionLimit
	}
	expires := now.Add(ttl)
	return Session{ID: id, VRAMLimitBytes: limitBytes, CreatedAt: now, UpdatedAt: now, ExpiresAt: expires, State: SessionActive}, nil
}

func (s Session) IsExpired(now time.Time) bool {
	return s.State == SessionExpired || now.After(s.ExpiresAt)
}
