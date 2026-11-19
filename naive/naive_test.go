package naive

import (
	"context"
	"strings"
	"testing"
)

var (
	ham  = "The quick brown fox jumps over the lazy dog"
	spam = "Earn cash quick online"
)

func TestProbability(t *testing.T) {
	classifier := New()

	t.Run(`Probabilities`, func(t *testing.T) {
		for z := 0; z < 1; z++ {
			classifier.TrainString(context.Background(), `aaa bbb ccc ddd`, "A")
			classifier.TrainString(context.Background(), `111 222 333 444 zzz`, "X")
			classifier.TrainString(context.Background(), `bbb ccc ddd eee`, "A")
			classifier.TrainString(context.Background(), `222 333 444 555 zzz`, "X")
			classifier.TrainString(context.Background(), `bbb ccc ddd eee fff`, "A")
			classifier.TrainString(context.Background(), `222 333 444 555 666 zzz`, "X")
		}

		if m, _ := classifier.Probabilities(`bbb ccc ddd`); m[`A`] <= m[`X`] {
			t.Errorf(`A=%.2f value should be greater than X=%.2f`, m[`X`], m[`A`])
		}

		if m, _ := classifier.Probabilities(`222 333 zzz`); m[`X`] <= m[`A`] {
			t.Errorf(`X=%.2f value should be greater than A=%.2f`, m[`X`], m[`A`])
		}
	})
}
func TestAddFeature(t *testing.T) {
	classifier := New()
	classifier.addFeature("quick", "good")
	assertFeatureCount(t, classifier, "quick", "good", 1.0)
	assertFeatureCount(t, classifier, "quick", "bad", 0.0)
	classifier.addFeature("quick", "bad")
	assertFeatureCount(t, classifier, "quick", "bad", 1.0)
}

func TestAddCategory(t *testing.T) {
	classifier := New()

	assertCategoryCount(t, classifier, "good", 0.0)
	classifier.addCategory("good")
	assertCategoryCount(t, classifier, "good", 1.0)
	categories := classifier.categories()

	assertEqual(t, float64(classifier.count()), float64(len(categories)))
}

func TestTrain(t *testing.T) {
	classifier := New()

	if err := classifier.TrainString(context.Background(), ham, "good"); err != nil {
		t.Error("classifier training failed")
	}

	if err := classifier.TrainString(context.Background(), spam, "bad"); err != nil {
		t.Error("classifier training failed")
	}

	assertFeatureCount(t, classifier, "quick", "good", 1.0)
	assertFeatureCount(t, classifier, "quick", "bad", 1.0)
	assertCategoryCount(t, classifier, "good", 1)
	assertCategoryCount(t, classifier, "bad", 1)
}

func TestClassify(t *testing.T) {
	classifier := New()
	text := "Quick way to make cash"

	t.Run("Empty classifier", func(t *testing.T) {
		if _, err := classifier.ClassifyString(context.Background(), text); err != ErrNotClassified {
			t.Errorf("expected classification error; received: %v", err)
		}
	})

	t.Run("Trained classifier", func(t *testing.T) {
		classifier.TrainString(context.Background(), ham, "good")
		classifier.TrainString(context.Background(), spam, "bad")

		if _, err := classifier.ClassifyString(context.Background(), text); err != nil {
			t.Error("document incorrectly classified")
		}
	})
}

func assertCategoryCount(t *testing.T, classifier *Classifier, category string, count float64) {
	v := classifier.categoryCount(category)
	assertEqual(t, count, v)
}

func assertFeatureCount(t *testing.T, classifier *Classifier, feature string, category string, count float64) {
	v := classifier.featureCount(feature, category)
	assertEqual(t, count, v)
}

func assertEqual(t *testing.T, expected, actual float64) {
	if actual != expected {
		t.Errorf("Expectation mismatch. Expected(%f) <=> Actual (%f)", expected, actual)
	}
}

func TestClassifyReader(t *testing.T) {
	c := New()
	c.TrainString(context.Background(), `aaa bbb ccc ddd`, "A")
	c.TrainString(context.Background(), `bbb ccc ddd eee`, "A")
	c.TrainString(context.Background(), `111 222 333 444 zzz`, "X")
	c.TrainString(context.Background(), `222 333 444 555 zzz`, "X")

	// map iteration order is random, so repeat to cover every category ordering
	for i := 0; i < 50; i++ {
		if got, err := c.Classify(context.Background(), strings.NewReader(`222 333 zzz`)); err != nil || got != "X" {
			t.Fatalf("expected X, got %q (err=%v)", got, err)
		}
	}
}
