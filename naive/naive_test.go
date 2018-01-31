package naive

import (
	"testing"
)

var (
	ham  = "The quick brown fox jumps over the lazy dog"
	spam = "Earn cash quick online"
)

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

	if err := classifier.TrainString(ham, "good"); err != nil {
		t.Error("classifier training failed")
	}

	if err := classifier.TrainString(spam, "bad"); err != nil {
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
		if _, err := classifier.ClassifyString(text); err != ErrNotClassified {
			t.Errorf("expected classification error; received: %v", err)
		}
	})

	t.Run("Trained classifier", func(t *testing.T) {
		classifier.TrainString(ham, "good")
		classifier.TrainString(spam, "bad")

		if _, err := classifier.ClassifyString(text); err != nil {
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
