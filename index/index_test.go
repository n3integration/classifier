package index

import (
	"strings"
	"testing"
)

var (
	text     = "The quick brown fox jumped over the lazy dog"
	expected = 7
)

func TestTermIndex(t *testing.T) {
	allTermsExpected := expected + 1
	index := NewTermIndex(allTermsExpected)
	for _, txt := range strings.Split(text, " ") {
		index.Add(strings.ToLower(txt))
	}
	if index.Count() != allTermsExpected {
		t.Errorf("incorrect index size; expected %v, but got %v", expected, index.Count())
	}
	for term := range index.terms {
		if index.Frequency(term) < 1 {
			t.Errorf("incorrect frequency; expected %v, but got %v", expected, index.Frequency(term))
		}
	}
}
