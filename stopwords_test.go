package classifier

import "testing"

func TestStopWords(t *testing.T) {
	t.Run("Stopword", func(t *testing.T) {
		sample := []string{"a", "is", "the"}
		for _, v := range sample {
			if IsNotStopWord(v) {
				t.Errorf("%s was not identified as a stop word", v)
			}
		}
	})
	t.Run("Other", func(t *testing.T) {
		sample := []string{"hello", "world"}
		for _, v := range sample {
			if IsStopWord(v) {
				t.Errorf("%s was incorrectly identified as a stop word", v)
			}
		}
	})
}
