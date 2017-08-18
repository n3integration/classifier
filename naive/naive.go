package naive

import (
	"errors"
	"sync"

	"github.com/n3integration/classifier"
)

// ErrNotClassified indicates that a document could not be classified
var ErrNotClassified = errors.New("unable to classify document")

// Classifier implements a naive bayes classifier
type Classifier struct {
	feat2cat map[string]map[string]int
	catCount map[string]int
	sync.RWMutex
}

// New initializes a new naive Classifier
func New() *Classifier {
	return &Classifier{
		feat2cat: make(map[string]map[string]int),
		catCount: make(map[string]int),
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
	max := 0.0
	var err error
	classification := ""
	probabilities := make(map[string]float64)

	c.RLock()
	defer c.RUnlock()

	for _, category := range c.categories() {
		if probabilities[category], err = c.probability(doc, category); err != nil {
			return "", err
		}
		if probabilities[category] > max {
			max = probabilities[category]
			classification = category
		}
	}
	if classification == "" {
		return "", ErrNotClassified
	}
	return classification, err
}

func (c *Classifier) addFeature(feature string, category string) {
	if _, ok := c.feat2cat[feature]; !ok {
		c.feat2cat[feature] = make(map[string]int)
	}
	c.feat2cat[feature][category]++
}

func (c *Classifier) featureCount(feature string, category string) float64 {
	if _, ok := c.feat2cat[feature]; ok {
		return float64(c.feat2cat[feature][category])
	}
	return 0.0
}

func (c *Classifier) addCategory(category string) {
	c.catCount[category]++
}

func (c *Classifier) categoryCount(category string) float64 {
	if _, ok := c.catCount[category]; ok {
		return float64(c.catCount[category])
	}
	return 0.0
}

func (c *Classifier) count() int {
	sum := 0
	for _, value := range c.catCount {
		sum += value
	}
	return sum
}

func (c *Classifier) categories() []string {
	var keys []string
	for k := range c.catCount {
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
