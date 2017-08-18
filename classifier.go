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

// Tokenize extracts and normalizes all words from a text corpus
func Tokenize(doc string) ([]string, error) {
	tokenizer, err := regexp.Compile("\\W+")

	if err != nil {
		return nil, err
	}

	tokens := tokenizer.Split(doc, -1)
	return Map(Filter(tokens, IsNotStopWord), strings.ToLower), nil
}

// WordCounts extracts term frequencies from a text corpus
func WordCounts(doc string) (map[string]int, error) {
	tokens, err := Tokenize(doc)

	if err != nil {
		return nil, err
	}

	wc := make(map[string]int)
	for _, token := range tokens {
		wc[token] = wc[token] + 1
	}

	return wc, nil
}
