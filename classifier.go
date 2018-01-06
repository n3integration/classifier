package classifier

import (
	"regexp"
	"strings"
)

// Classifier provides a simple interface for different text classifiers
type Classifier interface {
	// Train allows clients to train the classifier
	Train(doc string, category string) error
	// Classify performs a classification on the input corpus and assumes that
	// the underlying classifier has been trained.
	Classify(doc string) (string, error)
}

// WordCounts extracts term frequencies from a text corpus
func WordCounts(doc string) (map[string]int, error) {
	tokens := Tokenize(doc)
	wc := make(map[string]int)
	for _, token := range tokens {
		wc[token] = wc[token] + 1
	}

	return wc, nil
}
