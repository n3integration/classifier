package knn

import (
	"math"
)

// Similarity provides support for row similarity
type SimilarityScore func(left, right *row) float64

// EuclideanDistance between rows
func EuclideanDistance(left, right *row) float64 {
	distanceTo := func(left, right *row) float64 {
		score := 0.0
		terms := make(map[int]float64)
		for i := 0; i < left.Len(); i++ {
			term, val := left.Column(i)
			terms[term] = val
			score += math.Pow(val-right.Value(term), 2)
		}

		for i := 0; i < right.Len(); i++ {
			term, _ := right.Column(i)
			if _, ok := terms[term]; !ok {
				score += math.Pow(0-right.Value(term), 2)
			}
		}
		return 1 / (1 + math.Sqrt(score))
	}

	if left.Len() >= right.Len() {
		return distanceTo(left, right)
	}
	return distanceTo(right, left)
}

// CosineSimilarity between rows
func CosineSimilarity(left, right *row) float64 {
	return left.Dot(right) / (left.L2Norm() * right.L2Norm())
}

// PearsonCorrelation similarity between rows
func PearsonCorrelation(left, right *row) float64 {
	score := func(left, right *row) float64 {
		n := left.Size()
		leftSum := left.Sum()
		rightSum := right.Sum()
		denom := math.Sqrt((left.Square() - math.Pow(leftSum, 2)/n) * (right.Square() - math.Pow(rightSum, 2)/n))

		if denom == 0 {
			return 0
		}
		return (left.Dot(right) - ((leftSum * rightSum) / n)) / denom
	}

	similar := make([]int, 0)
	for i := 0; i < left.Len(); i++ {
		term, _ := left.Column(i)
		if right.Contains(term) {
			similar = append(similar, term)
		}
	}

	if len(similar) == 0 {
		return 0
	}
	return score(left.Values(similar...), right.Values(similar...))
}
