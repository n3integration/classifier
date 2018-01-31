package classifier

import (
	"bytes"
	"io"
	"testing"
)

var (
	text     = "The quick brown fox jumped over the lazy dog"
	expected = 7
)

func TestTokenize(t *testing.T) {
	t.Run("Standard Tokenizer", func(t *testing.T) {
		tokens := NewTokenizer().Tokenize(toReader(text))
		doTokenizeTest(t, tokens)
	})
	t.Run("Regexp Tokenizer", func(t *testing.T) {
		tokens := NewRegexTokenizer().Tokenize(toReader(text))
		doTokenizeTest(t, tokens)
	})
}

func toReader(text string) io.Reader {
	return bytes.NewBuffer([]byte(text))
}

func doTokenizeTest(t *testing.T, tokens chan string) {
	actual := 0
	for _ = range tokens {
		actual++
	}
	if actual != expected {
		t.Errorf("Expected %d tokens; actual: %d", expected, actual)
	}
}
