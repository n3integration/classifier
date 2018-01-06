package classifier

import (
	"regexp"
	"strings"
)

// Tokenizer provides a common interface to tokenize documents
type Tokenizer interface {
	// Tokenize breaks the provided document into a token slice
	Tokenize(doc string) []string
}

type stdTokenizer struct {
	tokenizer *regexp.Regexp
}

// NewTokenizer initializes a new standard Tokenizer instance
func NewTokenizer() Tokenizer {
	return &stdTokenizer{
		tokenizer: regexp.MustCompile("\\W+"),
	}
}

// Tokenize extracts and normalizes all words from a text corpus
func (t *stdTokenizer) Tokenize(doc string) []string {
	tokens := t.Split(doc, -1)
	return Map(Filter(tokens, IsNotStopWord), strings.ToLower)
}
