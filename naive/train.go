package naive

import (
	"context"
	"io"

	"github.com/n3integration/classifier"
)

// Train provides supervisory training to the classifier. Cancelling
// ctx aborts training without recording the document.
func (c *Classifier) Train(ctx context.Context, r io.Reader, category string) error {
	if err := validateCategory(category); err != nil {
		return err
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	err := classifier.ForEach(ctx, c.tokenizer.Tokenize(ctx, r), func(feature string) {
		c.addFeature(feature, category)
	})
	if err != nil {
		return err
	}

	c.addCategory(category)
	return nil
}

// TrainString provides supervisory training to the classifier
func (c *Classifier) TrainString(ctx context.Context, doc string, category string) error {
	return c.Train(ctx, asReader(doc), category)
}

func (c *Classifier) addFeature(feature string, category string) {
	if _, ok := c.feat2cat[feature]; !ok {
		c.feat2cat[feature] = make(map[string]int)
	}
	c.feat2cat[feature][category]++
}

func (c *Classifier) addCategory(category string) {
	c.catCount[category]++
}

func (c *Classifier) categories() []string {
	keys := make([]string, 0, len(c.catCount))
	for k := range c.catCount {
		keys = append(keys, k)
	}
	return keys
}
