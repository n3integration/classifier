package naive

import (
	"errors"
	"sync"

	"github.com/n3integration/classifier"
)

// Classifier implements a naive bayes classifier
type Classifier struct {
	f2c map[string]map[string]int
	cc  map[string]int
	sync.RWMutex
}

// New initializes a new naive Classifier
func New() *Classifier {
	return &Classifier{
		f2c: make(map[string]map[string]int),
		cc:  make(map[string]int),
	}
}

// Train provides supervisory training to the classifier
func (c *Classifier) Train(doc string, category string) error {
	features, err := classifier.Tokenize(doc)
	if err != nil {
		return err
	}

	c.Lock()
	defer c.Unlock()

	for _, feature := range features {
		c.addFeature(feature, category)
	}
	c.addCategory(category)
	return nil
}

// Classify attempts to classify a document. If the document cannot be classified
// (eg. because the classifier has not been trained), an error is returned.
func (c *Classifier) Classify(doc string) (string, error) {
	var err error
	max := 0.0
	classification := ""
	probabilities := make(map[string]float64)

	c.RLock()
	defer c.RUnlock()

	for _, category := range c.categories() {
		probabilities[category], err = c.probability(doc, category)
		if err != nil {
			return "", err
		}
		if probabilities[category] > max {
			max = probabilities[category]
			classification = category
		}
	}
	if classification == "" {
		return "", errors.New("unable to classify ")
	}
	return classification, nil
}

func (c *Classifier) addFeature(feature string, category string) {
	if _, ok := c.f2c[feature]; !ok {
		c.f2c[feature] = make(map[string]int)
	}
	c.f2c[feature][category]++
}

func (c *Classifier) featureCount(feature string, category string) float64 {
	if _, ok := c.f2c[feature]; ok {
		return float64(c.f2c[feature][category])
	}
	return 0.0
}

func (c *Classifier) addCategory(category string) {
	c.cc[category]++
}

func (c *Classifier) categoryCount(category string) float64 {
	if _, ok := c.cc[category]; ok {
		return float64(c.cc[category])
	}
	return 0.0
}

func (c *Classifier) count() int {
	sum := 0
	for _, value := range c.cc {
		sum += value
	}
	return sum
}

func (c *Classifier) categories() []string {
	var keys []string
	for k := range c.cc {
		keys = append(keys, k)
	}
	return keys
}

func (c *Classifier) featureProbability(feature string, category string) float64 {
	if c.categoryCount(category) == 0 {
		return 0.0
	}
	return c.featureCount(feature, category) / c.categoryCount(category)
}

func (c *Classifier) weightedProbability(feature string, category string) float64 {
	return c.variableWeightedProbability(feature, category, 1.0, 0.5)
}

func (c *Classifier) variableWeightedProbability(feature string, category string, weight float64, assumedProb float64) float64 {
	sum := 0.0
	probability := c.featureProbability(feature, category)
	for _, category := range c.categories() {
		sum += c.featureCount(feature, category)
	}
	return ((weight * assumedProb) + (sum * probability)) / (weight + sum)
}

func (c *Classifier) probability(doc string, category string) (float64, error) {
	categoryProbability := c.categoryCount(category) / float64(c.count())
	docProbability, err := c.docProbability(doc, category)
	if err != nil {
		return 0.0, nil
	}
	return docProbability * categoryProbability, nil
}

func (c *Classifier) docProbability(doc string, category string) (float64, error) {
	features, err := classifier.Tokenize(doc)
	if err != nil {
		return 0.0, err
	}
	probability := 1.0
	for _, feature := range features {
		probability *= c.weightedProbability(feature, category)
	}
	return probability, nil
}
