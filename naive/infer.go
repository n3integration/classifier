package naive

import (
	"context"
	"io"

	"github.com/n3integration/classifier"
)

// Classify attempts to classify a document. If the document cannot be classified
// (eg. because the classifier has not been trained), an error is returned, as is
// ctx's error if it is cancelled first.
func (c *Classifier) Classify(ctx context.Context, r io.Reader) (string, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	if len(c.catCount) == 0 {
		return "", ErrNotClassified
	}

	features, err := c.features(ctx, r)
	if err != nil {
		return "", err
	}

	maxProbability := 0.0
	classification := ""

	for _, category := range c.categories() {
		p := c.probability(features, category)
		if p > maxProbability {
			maxProbability = p
			classification = category
		}
	}

	if classification == "" {
		return "", ErrNotClassified
	}
	return classification, nil
}

// Probabilities runs the provided string through the model and returns
// the potential probability for each classification
func (c *Classifier) Probabilities(str string) (map[string]float64, string) {
	probabilities := make(map[string]float64)

	c.mu.RLock()
	defer c.mu.RUnlock()

	features, _ := c.features(context.Background(), asReader(str))
	cat := ``
	best := 0.0

	for _, category := range c.categories() {
		prob := c.probability(features, category)
		if prob > 0 {
			probabilities[category] = prob
		}
		if prob > best {
			best = prob
			cat = category
		}
	}

	return probabilities, cat
}

// ClassifyString provides convenience classification for strings
func (c *Classifier) ClassifyString(ctx context.Context, doc string) (string, error) {
	return c.Classify(ctx, asReader(doc))
}

func (c *Classifier) featureCount(feature string, category string) float64 {
	if _, ok := c.feat2cat[feature]; ok {
		return float64(c.feat2cat[feature][category])
	}
	return 0.0
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
	for _, _category := range c.categories() {
		sum += c.featureCount(feature, _category)
	}
	return ((weight * assumedProb) + (sum * probability)) / (weight + sum)
}

// features tokenizes a document once so that it can be scored against every category
func (c *Classifier) features(ctx context.Context, r io.Reader) ([]string, error) {
	var features []string
	err := classifier.ForEach(ctx, c.tokenizer.Tokenize(ctx, r), func(feature string) {
		features = append(features, feature)
	})
	return features, err
}

func (c *Classifier) probability(features []string, category string) float64 {
	categoryProbability := c.categoryCount(category) / float64(c.count())
	return c.docProbability(features, category) * categoryProbability
}

func (c *Classifier) docProbability(features []string, category string) float64 {
	probability := 1.0
	for _, feature := range features {
		probability *= c.weightedProbability(feature, category)
	}
	return probability
}
