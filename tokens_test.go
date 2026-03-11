package classifier

import (
	"bytes"
	"context"
	"io"
	"strings"
	"testing"
	"time"
	"unicode"
)

var (
	text     = "The quick brown fox jumped over the lazy dog"
	expected = 7
)

type assertion func(t *testing.T, v string)

func TestTokenize(t *testing.T) {
	tests := []struct {
		Name       string
		Opts       []StdOption
		Assertions []assertion
	}{
		{"Standard Tokenizer", options(), assertions()},
		{"Buffered Tokenizer", options(BufferSize(1)), assertions()},
		{"ToUpper Tokenizer", options(Transforms(toUpper)), assertions(isUpper)},
		{"Stopword Tokenizer", options(Filters(IsNotStopWord)), assertions(isStopWord)},
	}

	for _, test := range tests {
		t.Run(test.Name, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			tokens := NewTokenizer(test.Opts...).Tokenize(ctx, toReader(text))
			doTokenizeTest(t, tokens)
		})
	}
}

func isStopWord(t *testing.T, v string) {
	if IsStopWord(v) {
		t.Errorf("value is a stopword")
	}
}

func isUpper(t *testing.T, v string) {
	for _, c := range v {
		if !unicode.IsUpper(c) {
			t.Errorf("value is not in uppercase")
			return
		}
	}
}

func toUpper(s string) string {
	return strings.ToUpper(s)
}

func toReader(text string) io.Reader {
	return bytes.NewBuffer([]byte(text))
}

func doTokenizeTest(t *testing.T, tokens chan string, assertions ...assertion) {
	actual := 0
	for v := range tokens {
		for _, assert := range assertions {
			assert(t, v)
		}
		actual++
	}
	if actual != expected {
		t.Errorf("Expected %d tokens; actual: %d", expected, actual)
	}
}

func options(opts ...StdOption) []StdOption {
	return opts
}

func assertions(assertions ...assertion) []assertion {
	return assertions
}
