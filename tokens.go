package classifier

import (
	"bufio"
	"bytes"
	"io"
	"regexp"
	"strings"
	"unsafe"
)

// Tokenizer provides a common interface to tokenize documents
type Tokenizer interface {
	// Tokenize breaks the provided document into a token slice
	Tokenize(r io.Reader) chan string
}

type regexTokenizer struct {
	tokenizer *regexp.Regexp
}

type stdTokenizer struct {
}

// NewTokenizer initializes a new standard Tokenizer instance
func NewTokenizer() Tokenizer {
	return &stdTokenizer{}
}

// NewRegexTokenizer initializes a new regular expression Tokenizer instance
func NewRegexTokenizer() Tokenizer {
	return &regexTokenizer{
		tokenizer: regexp.MustCompile("\\W+"),
	}
}

func (t *stdTokenizer) Tokenize(r io.Reader) chan string {
	tokenizer := bufio.NewScanner(r)
	tokenizer.Split(bufio.ScanWords)
	tokens := make(chan string)

	go func() {
		for tokenizer.Scan() {
			tokens <- tokenizer.Text()
		}
		close(tokens)
	}()

	return pipeline(tokens)
}

// Tokenize extracts and normalizes all words from a text corpus
func (t *regexTokenizer) Tokenize(r io.Reader) chan string {
	buffer := new(bytes.Buffer)
	buffer.ReadFrom(r)
	b := buffer.Bytes()
	doc := *(*string)(unsafe.Pointer(&b))
	tokens := make(chan string)

	go func() {
		for _, token := range t.tokenizer.Split(doc, -1) {
			tokens <- token
		}
		close(tokens)
	}()

	return pipeline(tokens)
}

func pipeline(tokens chan string) chan string {
	return Map(Filter(tokens, IsNotStopWord), strings.ToLower)
}
