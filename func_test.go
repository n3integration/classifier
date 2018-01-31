package classifier

import (
	"strings"
	"testing"
)

var words = []string{
	"hello", "world",
}

func streamWords() chan string {
	stream := make(chan string)
	go func() {
		for _, word := range words {
			stream <- word
		}
		close(stream)
	}()
	return stream
}

func TestMap(t *testing.T) {
	i := 0
	results := Map(streamWords(), strings.ToUpper)
	for word := range results {
		expected := strings.ToUpper(words[i])
		if expected != word {
			t.Errorf("did not match expected result %v <> %v", expected, word)
		}
		i++
	}
}

func TestFilter(t *testing.T) {
	results := Filter(streamWords(), func(s string) bool {
		return s != words[0]
	})

	i := 0
	for word := range results {
		i++
		if word != words[1] {
			t.Error("incorrect result:", word)
		}
	}
	if i != 1 {
		t.Error("incorrect number of results:", i)
	}
}
