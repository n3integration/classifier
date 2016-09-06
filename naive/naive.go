package naive

import (
	"errors"
	"github.com/n3integration/classifier"
	"sync"
)

// Naive bayes classifier
type NaiveClassifier struct {
	f2c  map[string]map[string]int // feature to category counts
	cc   map[string]int            // category count
	lock sync.RWMutex
}

func NewClassifier() classifier.Classifier {
	return &NaiveClassifier{
		f2c: make(map[string]map[string]int),
		cc:  make(map[string]int),
	}
}

// Trains the classifier
func (this NaiveClassifier) Train(doc string, category string) error {
	features, err := classifier.Tokenize(doc)
	if err != nil {
		return err
	}

	this.lock.Lock()
	defer this.lock.Unlock()

	for _, feature := range features {
		this.addFeature(feature, category)
	}
	this.addCategory(category)
	return nil
}

// Classifies a document. If the document cannot be classified (eg. because
// the classifier has not been trained), an error is returned.
func (this NaiveClassifier) Classify(doc string) (string, error) {
	var err error
	max := 0.0
	classification := ""
	probabilities := make(map[string]float64)

	this.lock.RLock()
	defer this.lock.RUnlock()

	for _, category := range this.categories() {
		probabilities[category], err = this.probability(doc, category)
		if err != nil {
			return "", err
		}
		if probabilities[category] > max {
			max = probabilities[category]
			classification = category
		}
	}
	if classification == "" {
		return "", errors.New("classification unknown")
	}
	return classification, nil
}

func (this *NaiveClassifier) addFeature(feature string, category string) {
	if _, ok := this.f2c[feature]; !ok {
		this.f2c[feature] = make(map[string]int)
	}
	this.f2c[feature][category] += 1
}

func (this *NaiveClassifier) featureCount(feature string, category string) float64 {
	if _, ok := this.f2c[feature]; ok {
		return float64(this.f2c[feature][category])
	}
	return 0.0
}

func (this *NaiveClassifier) addCategory(category string) {
	this.cc[category] += 1
}

func (this *NaiveClassifier) categoryCount(category string) float64 {
	if _, ok := this.cc[category]; ok {
		return float64(this.cc[category])
	}
	return 0.0
}

func (this *NaiveClassifier) count() int {
	sum := 0
	for _, value := range this.cc {
		sum += value
	}
	return sum
}

func (this *NaiveClassifier) categories() []string {
	var keys []string
	for k := range this.cc {
		keys = append(keys, k)
	}
	return keys
}

func (this *NaiveClassifier) featureProbability(feature string, category string) float64 {
	if this.categoryCount(category) == 0 {
		return 0.0
	}
	return this.featureCount(feature, category) / this.categoryCount(category)
}

func (this *NaiveClassifier) weightedProbability(feature string, category string) float64 {
	return this.weightedProbabilityExt(feature, category, 1.0, 0.5)
}

func (this *NaiveClassifier) weightedProbabilityExt(feature string, category string, weight float64, assumedProb float64) float64 {
	sum := 0.0
	probability := this.featureProbability(feature, category)
	for _, category := range this.categories() {
		sum += this.featureCount(feature, category)
	}
	return ((weight * assumedProb) + (sum * probability)) / (weight + sum)
}

func (this *NaiveClassifier) probability(doc string, category string) (float64, error) {
	categoryProbability := this.categoryCount(category) / float64(this.count())
	docProbability, err := this.docProbability(doc, category)
	if err != nil {
		return 0.0, nil
	}
	return docProbability * categoryProbability, nil
}

func (this *NaiveClassifier) docProbability(doc string, category string) (float64, error) {
	features, err := classifier.Tokenize(doc)
	if err != nil {
		return 0.0, err
	}
	probability := 1.0
	for _, feature := range features {
		probability *= this.weightedProbability(feature, category)
	}
	return probability, nil
}
