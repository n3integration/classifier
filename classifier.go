package classifier

import (
	"regexp"
	"strings"
)

type Classifier interface {
	Train(doc string, category string) error
	Classify(doc string) (string, error)
}

// Predicate to determine whether or not a token should be considered a
// valid word (i.e. greater than two but less than 20 characters).
func IsAWord(v string) bool {
	return len(v) > 2 && len(v) < 20
}

// Extracts and normalizes all words from a text corpus
func Tokenize(doc string) ([]string, error) {
	tokenizer, err := regexp.Compile("\\W+")

	if err != nil {
		return nil, err
	}

	tokens := tokenizer.Split(doc, -1)
	return Map(Filter(tokens, IsAWord), strings.ToLower), nil
}

// Extracts term frequencies from a text corpus
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

// Applies f to each element of vs
func Map(vs []string, f func(string) string) []string {
	vsm := make([]string, len(vs))
	for i, v := range vs {
		vsm[i] = f(v)
	}
	return vsm
}

// Removes elements from vs where the predicate is satisfied
func Filter(vs []string, f func(string) bool) []string {
	vsf := make([]string, 0)
	for _, v := range vs {
		if f(v) {
			vsf = append(vsf, v)
		}
	}
	return vsf
}
