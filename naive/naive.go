package naive

import (
	"bytes"
	"errors"
	"io"
	"strings"
	"sync"
	"unicode/utf8"

	"github.com/n3integration/classifier"
)

const maxCategorySize = 256

// ErrNotClassified indicates that a document could not be classified
var ErrNotClassified = errors.New("unable to classify document")

// ErrInvalidCategory is returned when a category name fails validation
var ErrInvalidCategory = errors.New("invalid category: must be non-empty and no longer than 256 characters")

// Option provides a functional setting for the Classifier
type Option func(c *Classifier) error

// Classifier implements a naive bayes classifier
type Classifier struct {
	feat2cat  map[string]map[string]int
	catCount  map[string]int
	tokenizer classifier.Tokenizer
	mu        sync.RWMutex
}

// New initializes a new naive Classifier using the standard tokenizer
func New(opts ...Option) *Classifier {
	c := &Classifier{
		feat2cat:  make(map[string]map[string]int),
		catCount:  make(map[string]int),
		tokenizer: classifier.NewTokenizer(),
	}
	for _, opt := range opts {
		if err := opt(c); err != nil {
			panic(err)
		}
	}
	return c
}

func validateCategory(category string) error {
	if len(strings.TrimSpace(category)) == 0 {
		return ErrInvalidCategory
	}
	if utf8.RuneCountInString(category) > maxCategorySize {
		return ErrInvalidCategory
	}
	return nil
}

// Tokenizer overrides the classifier's default Tokenizer
func Tokenizer(t classifier.Tokenizer) Option {
	return func(c *Classifier) error {
		c.tokenizer = t
		return nil
	}
}

func asReader(text string) io.Reader {
	return bytes.NewBufferString(text)
}
