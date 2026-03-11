package classifier

import (
	"bufio"
	"context"
	"io"
	"strings"
	"unicode"
	"unicode/utf8"
)

// Tokenizer provides a common interface to tokenize documents
type Tokenizer interface {
	// Tokenize breaks the provided document into a channel of tokens
	Tokenize(context.Context, io.Reader) chan string
}

// IsWord is a predicate to determine if a string contains at least two
// characters and doesn't contain any numbers
func IsWord(v string) bool {
	if len(v) < 2 {
		return false
	}
	for _, r := range v {
		if unicode.IsLetter(r) {
			return true
		}
	}
	return false
}

// StdOption provides configuration settings for a StdTokenizer
type StdOption func(*StdTokenizer)

// StdTokenizer provides a common document tokenizer that splits a
// document by word boundaries
type StdTokenizer struct {
	transforms []Mapper
	splitFn    bufio.SplitFunc
	filters    []Predicate
	bufferSize int
}

// NewTokenizer initializes a new standard Tokenizer instance
func NewTokenizer(opts ...StdOption) *StdTokenizer {
	tokenizer := &StdTokenizer{
		bufferSize: 100,
		splitFn:    bufio.ScanWords,
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

// Tokenize words and return streaming results
func (t *StdTokenizer) Tokenize(ctx context.Context, r io.Reader) chan string {
	ch := make(chan string, t.bufferSize)

	go func() {
		defer close(ch)

		scanner := bufio.NewScanner(r)
		scanner.Split(t.splitFn)

		for scanner.Scan() {
			token := scanner.Text()

			// Apply transforms
			for _, m := range t.transforms {
				token = m(token)
			}

			// Apply filter predicates
			keep := true
			for _, f := range t.filters {
				if !f(token) {
					keep = false
					break
				}
			}
			if !keep {
				continue
			}

			select {
			case ch <- token:
			case <-ctx.Done():
				return
			}
		}
	}()
	return ch
}

func (t *StdTokenizer) pipeline(in chan string) chan string {
	return Map(Filter(in, t.filters...), t.transforms...)
}

// BufferSize adjusts the size of the buffered channel
func BufferSize(size int) StdOption {
	return func(t *StdTokenizer) {
		if size < 1 {
			size = 1
		}
		t.bufferSize = size
	}
}

// SplitFunc overrides the default word split function, based on whitespace
func SplitFunc(fn bufio.SplitFunc) StdOption {
	return func(t *StdTokenizer) {
		t.splitFn = fn
	}
}

// Transforms overrides the list of mappers
func Transforms(m ...Mapper) StdOption {
	return func(t *StdTokenizer) {
		t.transforms = m
	}
}

// Filters overrides the list of predicates
func Filters(f ...Predicate) StdOption {
	return func(t *StdTokenizer) {
		t.filters = f
	}
}

// ScanAlphaWords is a function that splits text on whitespace, punctuation, and symbols;
// derived bufio.ScanWords
func ScanAlphaWords(data []byte, atEOF bool) (advance int, token []byte, err error) {
	// Skip leading spaces and symbols
	start := 0
	for width := 0; start < len(data); start += width {
		var r rune
		r, width = utf8.DecodeRune(data[start:])

		if !unicode.IsSpace(r) && !unicode.IsPunct(r) && !unicode.IsSymbol(r) {
			break
		}
	}

	// Scan until space or symbol, marking end of word.
	for width, i := 0, start; i < len(data); i += width {
		var r rune
		r, width = utf8.DecodeRune(data[i:])
		if unicode.IsSpace(r) || unicode.IsPunct(r) || unicode.IsSymbol(r) {
			return i + width, data[start:i], nil
		}
	}

	// If we're at EOF, we have a final, non-empty, non-terminated word. Return it.
	if atEOF && len(data) > start {
		return len(data), data[start:], nil
	}
	// Request more data.
	return start, nil, nil
}
