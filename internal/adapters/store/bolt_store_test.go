package store

import (
	"context"
	"gslice/internal/domain"
	"path/filepath"
	"testing"
	"time"
)

func TestBoltStoreCRUD(t *testing.T) {
	dir := t.TempDir()
	s, err := NewBoltStore(filepath.Join(dir, "test.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()

	session, _ := domain.NewSession("s1", 10, time.Now(), time.Minute)
	if err := s.Create(context.Background(), session); err != nil {
		t.Fatal(err)
	}
	got, err := s.Get(context.Background(), "s1")
	if err != nil || got.ID != "s1" {
		t.Fatalf("get failed %v", err)
	}
	if err := s.UpdateUsedBytes(context.Background(), "s1", 7); err != nil {
		t.Fatal(err)
	}
	got, _ = s.Get(context.Background(), "s1")
	if got.UsedBytes != 7 {
		t.Fatal("update failed")
	}
	if err := s.Delete(context.Background(), "s1"); err != nil {
		t.Fatal(err)
	}
}
