package classifier

import "testing"

var (
	text     = "The quick brown fox jumped over the lazy dog"
	expected = 7
)

func TestTokenize(t *testing.T) {
	tokens, err := Tokenize(text)

	if err != nil {
		t.Error("failed to tokenize text:", err)
	}

	if len(tokens) != expected {
		t.Errorf("Expected %d tokens; actual: %d", expected, len(tokens))
	}
}

func TestWordCounts(t *testing.T) {
	wc, err := WordCounts(text)

	if err != nil {
		t.Error("failed to get word counts:", err)
	}

	if len(wc) != expected {
		t.Errorf("Expected %d; actual %d", expected, len(wc))
	}

	for key, value := range wc {
		if value != 1 {
			t.Errorf("Incorrect term frequency for %s: %d", key, value)
		}
	}
}
