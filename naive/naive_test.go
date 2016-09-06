package naive

import "testing"

func NewNaiveClassifier() *NaiveClassifier {
	return &NaiveClassifier{
		f2c: make(map[string]map[string]int),
		cc:  make(map[string]int),
	}
}

func TestAddFeature(t *testing.T) {
	classifier := NewNaiveClassifier()
	classifier.addFeature("quick", "good")
	assertFeatureCount(t, classifier, "quick", "good", 1.0)
	assertFeatureCount(t, classifier, "quick", "bad", 0.0)
	classifier.addFeature("quick", "bad")
	assertFeatureCount(t, classifier, "quick", "bad", 1.0)
}

func assertFeatureCount(t *testing.T, classifier *NaiveClassifier, feature string, category string, count float64) {
	v := classifier.featureCount(feature, category)
	if v != count {
		t.Error("Expected ", count, ", got ", v)
	}
}

func TestAddCategory(t *testing.T) {
	classifier := NewNaiveClassifier()
	assertCategoryCount(t, classifier, "good", 0.0)
	classifier.addCategory("good")
	assertCategoryCount(t, classifier, "good", 1.0)
	categories := classifier.categories()
	if len(categories) != classifier.count() {
		t.Error("Expected ", classifier.count(), ", got ", len(categories))
	}
}

func assertCategoryCount(t *testing.T, classifier *NaiveClassifier, category string, count float64) {
	v := classifier.categoryCount(category)
	if v != count {
		t.Error("Expected ", count, ", got ", v)
	}
}

func TestTrain(t *testing.T) {
	classifier := NewNaiveClassifier()
	if err := classifier.Train("The quick brown fox jumps over the lazy dog", "good"); err != nil {
		t.Error("Classifier training failed")
	}
	if err := classifier.Train("Earn cash quick online", "bad"); err != nil {
		t.Error("Classifier training failed")
	}
	assertFeatureCount(t, classifier, "quick", "good", 1.0)
	assertFeatureCount(t, classifier, "quick", "bad", 1.0)
	assertCategoryCount(t, classifier, "good", 1)
	assertCategoryCount(t, classifier, "bad", 1)

	if _, err := classifier.Classify("Quick way to make cash"); err != nil {
		t.Error("Document incorrectly classified")
	}
}
