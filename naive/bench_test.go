package naive

import (
	"context"
	"fmt"
	"strings"
	"testing"
)

func benchClassifier(b *testing.B) *Classifier {
	c := New()
	for i := 0; i < 50; i++ {
		c.TrainString(context.Background(), strings.Repeat(fmt.Sprintf("alpha%d beta gamma ", i), 20), fmt.Sprintf("cat%d", i%10))
	}
	b.ResetTimer()
	return c
}

func BenchmarkClassify(b *testing.B) {
	c := benchClassifier(b)
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		c.ClassifyString(context.Background(), "alpha3 beta gamma delta")
	}
}

func BenchmarkClassifyParallel(b *testing.B) {
	c := benchClassifier(b)
	b.ReportAllocs()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			c.ClassifyString(context.Background(), "alpha3 beta gamma delta")
		}
	})
}
