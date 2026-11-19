package naive

import (
	"context"
	"errors"
	"io"
	"testing"
	"time"
)

// blockingReader never returns data until released, simulating slow I/O
type blockingReader struct{ done chan struct{} }

func (b blockingReader) Read([]byte) (int, error) {
	<-b.done
	return 0, io.EOF
}

func TestTrainCancelled(t *testing.T) {
	br := blockingReader{done: make(chan struct{})}
	defer close(br.done)

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	errc := make(chan error, 1)
	go func() { errc <- New().Train(ctx, br, "A") }()

	select {
	case err := <-errc:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("expected context.Canceled, got %v", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("Train did not abort on cancellation")
	}
}

func TestClassifyCancelled(t *testing.T) {
	c := New()
	c.TrainString(context.Background(), "aaa bbb", "A")

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := c.Classify(ctx, asReader("aaa bbb")); !errors.Is(err, context.Canceled) {
		t.Fatalf("expected context.Canceled, got %v", err)
	}
}
