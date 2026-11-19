package knn

import (
	"context"
	"fmt"
	"strings"
	"testing"
)

func benchKNN(b *testing.B) *Classifier {
	c := New(K(3))
	for i := 0; i < 200; i++ {
		c.TrainString(context.Background(), strings.Repeat(fmt.Sprintf("alpha%d beta gamma%d ", i, i%7), 10), fmt.Sprintf("cat%d", i%10))
	}
	b.ResetTimer()
	return c
}

func BenchmarkKNNClassify(b *testing.B) {
	c := benchKNN(b)
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		c.ClassifyString(context.Background(), "alpha3 beta gamma3 delta")
	}
}

func BenchmarkKNNClassifyParallel(b *testing.B) {
	c := benchKNN(b)
	b.ReportAllocs()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			c.ClassifyString(context.Background(), "alpha3 beta gamma3 delta")
		}
	})
}
