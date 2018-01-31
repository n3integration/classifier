package classifier

import (
	"testing"
)

func TestWordCounts(t *testing.T) {
	wc, err := WordCounts(toReader(text))

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
