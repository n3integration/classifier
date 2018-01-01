package classifier

import (
	"strings"
	"testing"
)

var words = []string{
	"hello", "world",
}

func TestMap(t *testing.T) {
	result := Map(words, strings.ToUpper)
	for i, word := range result {
		expected := strings.ToUpper(words[i])
		if expected != word {
			t.Errorf("did not match expected result %v <> %v", expected, word)
		}
	}
}

func TestFilter(t *testing.T) {
	result := Filter(words, func(s string) bool {
		return s != "hello"
	})

	if len(result) != 1 {
		t.Error("incorrect number of results:", len(result))
	}
	if result[0] != "world" {
		t.Error("incorrect result:", result[0])
	}
}
