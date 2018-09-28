package classifier

import (
	"bufio"
	"io"
	"strings"
)

// Tokenizer provides a common interface to tokenize documents
type Tokenizer interface {
	// Tokenize breaks the provided document into a channel of tokens
	Tokenize(io.Reader) chan string
}

// StdOption provides configuration settings for a StdTokenizer
type StdOption func(*StdTokenizer)

// StdTokenizer provides a common document tokenizer that splits a
// document by word boundaries
type StdTokenizer struct {
	transforms []Mapper
	filters    []Predicate
	bufferSize int
}

// NewTokenizer initializes a new standard Tokenizer instance
func NewTokenizer(opts ...StdOption) *StdTokenizer {
	tokenizer := &StdTokenizer{
		bufferSize: 100,
		transforms: []Mapper{
			strings.ToLower,
		},
		filters: []Predicate{
			IsNotStopWord,
		},
	}
	for _, opt := range opts {
		opt(tokenizer)
	}
	return tokenizer
}

func (t *StdTokenizer) Tokenize(r io.Reader) chan string {
	tokenizer := bufio.NewScanner(r)
	tokenizer.Split(bufio.ScanWords)
	tokens := make(chan string, t.bufferSize)

	go func() {
		for tokenizer.Scan() {
			tokens <- tokenizer.Text()
		}
		close(tokens)
	}()

	return t.pipeline(tokens)
}

func (t *StdTokenizer) pipeline(in chan string) chan string {
	return Map(Filter(in, t.filters...), t.transforms...)
}

func BufferSize(size int) StdOption {
	return func(t *StdTokenizer) {
		t.bufferSize = size
	}
}

func Transforms(m ...Mapper) StdOption {
	return func(t *StdTokenizer) {
		t.transforms = m
	}
}

func Filters(f ...Predicate) StdOption {
	return func(t *StdTokenizer) {
		t.filters = f
	}
}