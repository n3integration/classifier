package knn

import (
	"math"
	"testing"
)

func TestSimilarity(t *testing.T) {
	allowedVariance := .01
	row1 := newSparseRow(2)
	row1.ind = []int{0, 1}
	row1.val = []float64{2, -1}
	row2 := newSparseRow(2)
	row2.ind = []int{0, 1}
	row2.val = []float64{-2, 1}

	t.Run("Euclidean Distance", func(t *testing.T) {
		expected := 0.18
		actual := EuclideanDistance(row1, row2)
		assertEquivalent(t, expected, actual, allowedVariance)

		if actual := EuclideanDistance(row1, row1); actual != 1 {
			t.Fatalf("expected identical row to equal one. got %.2f", actual)
		}
	})

	t.Run("Pearson Correlation", func(t *testing.T) {
		if actual := PearsonCorrelation(row1, row1); actual != 1 {
			t.Fatalf("expected strong positive correlation. got %.2f", actual)
		}

		if actual := PearsonCorrelation(row1, row2); actual != -1 {
			t.Fatalf("expected strong inverse correlation. got %.2f", actual)
		}

		row3 := newSparseRow(2)
		row3.ind = []int{2, 3}
		row3.val = []float64{4, 5}
		if actual := PearsonCorrelation(row1, row3); actual != 0 {
			t.Fatalf("expected dissimilar rows to equal zero; got %.2f", actual)
		}
	})

	t.Run("Cosine Similarity", func(t *testing.T) {
		assertEquivalent(t, CosineSimilarity(row1, row1), 1.0, allowedVariance)
		assertEquivalent(t, CosineSimilarity(row1, row2), -1.0, allowedVariance)
	})
}

func assertEquivalent(t *testing.T, actual, expected, threshold float64) {
	if math.Abs(actual-expected) > threshold {
		t.Fatalf("expected %.2f to be equivalent to %.2f within +/- %.2f", actual, expected, threshold)
	}
}
