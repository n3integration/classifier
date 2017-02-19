package classifier

import (
	"sort"
	"strings"
)

var (
	stopwords = []string{
		"a", "able", "about", "across", "after", "all", "almost", "also", "am", "among", "an", "and", "any", "are", "as", "at",
		"be", "because", "been", "but", "by", "can", "cannot", "could", "dear", "did", "do", "does", "either", "else", "ever",
		"every", "for", "from", "get", "got", "had", "has", "have", "he", "her", "hers", "him", "his", "how", "however", "i",
		"if", "in", "into", "is", "it", "its", "just", "least", "let", "like", "likely", "may", "me", "might", "most", "must",
		"my", "neither", "no", "nor", "not", "of", "off", "often", "on", "only", "or", "other", "our", "own", "rather", "said",
		"say", "says", "she", "should", "since", "so", "some", "than", "that", "the", "their", "them", "then", "there", "these",
		"they", "this", "tis", "to", "too", "twas", "us", "wants", "was", "we", "were", "what", "when", "where", "which", "while",
		"who", "whom", "why", "will", "with", "would", "yet", "you", "your"}
	numStopwords = len(stopwords)
)

// IsStopWord performs a binary search against a list of known english stop words
// returns true if v is a stop word; false otherwise
func IsStopWord(v string) bool {
	v = strings.ToLower(v)
	index := sort.SearchStrings(stopwords, v)
	return index >= numStopwords || stopwords[index] == v
}

// IsNotStopWord is the inverse function of IsStopWord
func IsNotStopWord(v string) bool {
	return !IsStopWord(v)
}
