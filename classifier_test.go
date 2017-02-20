package classifier

import "testing"

var (
	text               = "The quick brown fox jumped over the lazy dog"
	expectedTokenCount = 7
)

func TestTokenize(t *testing.T) {
	tokens, err := Tokenize(text)

	if err != nil {
		t.Error("failed to tokenize text:", err)
	}

	if len(tokens) != expectedTokenCount {
		t.Error("Expected %d tokens; actual: %d", expected, len(tokens))
	}
}

func TestWordCounts(t *testing.T) {
	wc, err := WordCounts(text)

	if err != nil {
		t.Error("failed to get word counts:", err)
	}

	if len(wc) != expectedTokenCount {
		t.Error("Expected %d; actual %d", expected, wc)
	}

	for key, value := range wc {
		if value != 1 {
			t.Error("Incorrect term frequency for %s: %d", key, value)
		}
	}
}
