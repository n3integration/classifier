package naive

import (
	"bytes"
	"errors"
	"io"
	"sync"

	"github.com/n3integration/classifier"
)

// ErrNotClassified indicates that a document could not be classified
var ErrNotClassified = errors.New("unable to classify document")

// Option provides a functional setting for the Classifier
type Option func(c *Classifier) error

// Classifier implements a naive bayes classifier
type Classifier struct {
	feat2cat  map[string]map[string]int
	catCount  map[string]int
	tokenizer classifier.Tokenizer
	mu        sync.RWMutex
}

// New initializes a new naive Classifier using the standard tokenizer
func New(opts ...Option) *Classifier {
	c := &Classifier{
		feat2cat:  make(map[string]map[string]int),
		catCount:  make(map[string]int),
		tokenizer: classifier.NewTokenizer(),
	}
	for _, opt := range opts {
		opt(c)
	}
	return c
}

// Tokenizer overrides the classifier's default Tokenizer
func Tokenizer(t classifier.Tokenizer) Option {
	return func(c *Classifier) error {
		c.tokenizer = t
		return nil
	}
}

// Train provides supervisory training to the classifier
func (c *Classifier) Train(r io.Reader, category string) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	for feature := range c.tokenizer.Tokenize(r) {
		c.addFeature(feature, category)
	}

	c.addCategory(category)
	return nil
}

// TrainString provides supervisory training to the classifier
func (c *Classifier) TrainString(doc string, category string) error {
	return c.Train(asReader(doc), category)
}

// Classify attempts to classify a document. If the document cannot be classified
// (eg. because the classifier has not been trained), an error is returned.
func (c *Classifier) Classify(r io.Reader) (string, error) {
	max := 0.0
	var err error
	classification := ""
	probabilities := make(map[string]float64)

	c.mu.RLock()
	defer c.mu.RUnlock()

	for _, category := range c.categories() {
		probabilities[category] = c.probability(r, category)
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

// ClassifyString provides convenience classification for strings
func (c *Classifier) ClassifyString(doc string) (string, error) {
	return c.Classify(asReader(doc))
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

func (c *Classifier) probability(r io.Reader, category string) float64 {
	categoryProbability := c.categoryCount(category) / float64(c.count())
	docProbability := c.docProbability(r, category)
	return docProbability * categoryProbability
}

func (c *Classifier) docProbability(r io.Reader, category string) float64 {
	probability := 1.0
	for feature := range c.tokenizer.Tokenize(r) {
		probability *= c.weightedProbability(feature, category)
	}
	return probability
}

func asReader(text string) io.Reader {
	return bytes.NewBufferString(text)
}
