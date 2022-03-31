package naive

import (
	"fmt"
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
			classifier.TrainString(`aaa bbb ccc ddd`, "A")
			classifier.TrainString(`111 222 333 444 zzz`, "X")
			classifier.TrainString(`bbb ccc ddd eee`, "A")
			classifier.TrainString(`222 333 444 555 zzz`, "X")
			classifier.TrainString(`bbb ccc ddd eee fff`, "A")
			classifier.TrainString(`222 333 444 555 666 zzz`, "X")
		}

		if m, _ := classifier.Probabilities(`bbb ccc ddd`); m[`A`] <= m[`X`] {
			t.Errorf(`A=%.2f value should be greater than X=%.2f`, m[`X`], m[`A`])
		} else {
			fmt.Println(m)
		}

		if m, _ := classifier.Probabilities(`222 333 zzz`); m[`X`] <= m[`A`] {
			t.Errorf(`X=%.2f value should be greater than A=%.2f`, m[`X`], m[`A`])
		} else {
			fmt.Println(m)
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

		if cla, err := classifier.ClassifyString(ham); err != nil || cla != `good` {
			t.Error("document incorrectly classified: 'ham' should be 'good', but got " + cla)
		}
		if cla, err := classifier.ClassifyString(spam); err != nil || cla != `bad` {
			t.Error("document incorrectly classified: 'spam' should be 'bad', but got " + cla)
		}
		fmt.Println(classifier.Probabilities(ham))
		fmt.Println(classifier.Probabilities(spam))
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
