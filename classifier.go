package classifier

import (
	"context"
	"io"
)

// Classifier provides a simple interface for different text classifiers
type Classifier interface {
	// Train allows clients to train the classifier. Cancelling ctx aborts
	// training without recording the document.
	Train(ctx context.Context, r io.Reader, category string) error
	// TrainString allows clients to train the classifier using a string
	TrainString(ctx context.Context, doc string, category string) error
	// Classify performs a classification on the input corpus and assumes that
	// the underlying classifier has been trained. ctx's error is returned if
	// it is cancelled first.
	Classify(ctx context.Context, r io.Reader) (string, error)
	// ClassifyString performs text classification using a string
	ClassifyString(ctx context.Context, doc string) (string, error)
}

// WordCounts extracts term frequencies from a text corpus, returning
// ctx's error if it is cancelled before the corpus is fully read
func WordCounts(ctx context.Context, r io.Reader) (map[string]int, error) {
	wc := make(map[string]int)
	err := ForEach(ctx, NewTokenizer().Tokenize(ctx, r), func(token string) {
		wc[token]++
	})
	if err != nil {
		return nil, err
	}
	return wc, nil
}

// ForEach invokes fn for every token received from ch, returning ctx's error
// as soon as ctx is cancelled, even if the producer is blocked
func ForEach(ctx context.Context, ch <-chan string, fn func(string)) error {
	for {
		select {
		case token, ok := <-ch:
			if !ok {
				return ctx.Err()
			}
			fn(token)
		case <-ctx.Done():
			return ctx.Err()
		}
	}
}
