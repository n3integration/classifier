package knn

import (
	"math"
)

// WeightSchemeStrategy provides support for pluggable weight schemes
type WeightSchemeStrategy func(doc map[string]float64) weightScheme

// weightScheme provides a contract for termRef frequency weight schemes
type weightScheme func(term string) float64

// Binary weight scheme: 1 if present; 0 otherwise
func Binary(doc map[string]float64) weightScheme {
	return func(term string) float64 {
		if _, ok := doc[term]; ok {
			return 1
		}
		return 0
	}
}

// BagOfWords weight scheme: counts the number of occurrences
func BagOfWords(doc map[string]float64) weightScheme {
	return func(term string) float64 {
		return doc[term]
	}
}

// TermFrequency weight scheme; counts the number of occurrences divided by
// the number of terms within a document
func TermFrequency(doc map[string]float64) weightScheme {
	return func(term string) float64 {
		return math.Sqrt(doc[term] / float64(len(doc)))
	}
}

// LogNorm weight scheme: returns the natural log of the number of occurrences of a term
func LogNorm(doc map[string]float64) weightScheme {
	return func(term string) float64 {
		return math.Log(1 + doc[term])
	}
}
