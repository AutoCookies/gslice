package domain

import "testing"

func TestReserveAndRelease(t *testing.T) {
	used, err := Reserve(0, 100, 60)
	if err != nil || used != 60 {
		t.Fatalf("unexpected reserve result %d %v", used, err)
	}
	if _, err := Reserve(60, 100, 50); err != ErrQuotaExceeded {
		t.Fatalf("expected quota exceeded")
	}
	if got := Release(60, 10); got != 50 {
		t.Fatalf("release mismatch")
	}
	if got := Release(50, 100); got != 0 {
		t.Fatalf("release clamp mismatch")
	}
}
