package domain

import (
	"testing"
	"time"
)

func TestSessionLifecycle(t *testing.T) {
	now := time.Unix(100, 0)
	s, err := NewSession("abc", 1024, now, time.Second)
	if err != nil {
		t.Fatal(err)
	}
	if s.IsExpired(now) {
		t.Fatal("not expired yet")
	}
	if !s.IsExpired(now.Add(2 * time.Second)) {
		t.Fatal("expected expired")
	}
}
