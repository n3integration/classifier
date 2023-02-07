package knn

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"math"
	"sort"
	"sync"

	"github.com/n3integration/classifier"
	"github.com/n3integration/classifier/index"
)

const (
	defaultKVal          = 1
	defaultIndexCapacity = 10_000
)

// Option provides a functional setting for the Classifier
type Option func(c *Classifier) error

// Classifier provides k-nearest neighbor classification
type Classifier struct {
	mu sync.RWMutex

	k            int
	categories   []string
	index        *index.TermIndex
	matrix       *sparse
	similarity   SimilarityScore
	tokenizer    classifier.Tokenizer
	weightScheme classifier.WeightSchemeStrategy
}

// New initializes a new k-nearest neighbor classifier unless overridden,
// binary term weights and k=1 will be used for the created instance
func New(opts ...Option) *Classifier {
	c := &Classifier{
		k:            defaultKVal,
		categories:   make([]string, 0),
		index:        index.NewTermIndex(defaultIndexCapacity),
		matrix:       newSparseMatrix(),
		similarity:   CosineSimilarity,
		tokenizer:    classifier.NewTokenizer(),
		weightScheme: classifier.Binary,
	}
	for _, opt := range opts {
		opt(c)
	}
	return c
}

// K provides the value of 'k'
func K(k int) Option {
	return func(c *Classifier) error {
		if k < 1 {
			return errors.New("the value of k must be a positive integer")
		}
		c.k = k
		return nil
	}
}

// WeightScheme provides the term weight scheme
func WeightScheme(s classifier.WeightSchemeStrategy) Option {
	return func(c *Classifier) error {
		c.weightScheme = s
		return nil
	}
}

// Similarity provides an alternate similarity scoring strategy
func Similarity(s SimilarityScore) Option {
	return func(c *Classifier) error {
		c.similarity = s
		return nil
	}
}

// Tokenizer provides an alternate document Tokenizer
func Tokenizer(t classifier.Tokenizer) Option {
	return func(c *Classifier) error {
		c.tokenizer = t
		return nil
	}
}

// TermIndex provides an alternate TermIndex
func TermIndex(i *index.TermIndex) Option {
	return func(c *Classifier) error {
		c.index = i
		return nil
	}
}

func (c *Classifier) TrainString(doc string, category string) error {
	return c.Train(asReader(doc), category)
}

func (c *Classifier) Train(r io.Reader, category string) error {
	wordFreq := make(map[string]float64)
	for text := range c.tokenizer.Tokenize(r) {
		count := wordFreq[text]
		wordFreq[text] = count + 1

		if count == 0 {
			c.index.Add(text)
		}
	}

	c.mu.Lock()
	defer c.mu.Unlock()
	c.categories = append(c.categories, category)
	c.matrix.Add(c.index, c.weightScheme(wordFreq), wordFreq)
	return nil
}

func (c *Classifier) ClassifyString(doc string) (string, error) {
	return c.Classify(asReader(doc))
}

func (c *Classifier) Classify(r io.Reader) (string, error) {
	wordFreq := make(map[string]float64)
	for text := range c.tokenizer.Tokenize(r) {
		count := wordFreq[text]
		wordFreq[text] = count + 1
	}

	c.mu.RLock()
	defer c.mu.RUnlock()
	this := c.matrix.MakeRow(c.index, c.weightScheme, wordFreq)
	next := c.matrix.Rows()
	results := make(topResults, 0)

	for row := next(); row != nil; row = next() {
		results = append(results, &topResult{
			Score:    c.similarity(row, this),
			Category: c.categories[row.Index()],
		})
	}

	sort.Sort(results)
	return results.query(c.k), nil
}

type topResults []*topResult

func (r topResults) Len() int {
	return len(r)
}

func (r topResults) Less(i, j int) bool {
	return r[i].Score < r[j].Score
}

func (r topResults) Swap(i, j int) {
	r[i], r[j] = r[j], r[i]
}

func (r topResults) topK(k int) map[string]int {
	count := 0
	topk := make(map[string]int)
	for i := 1; i <= k; i++ {
		count = topk[r[len(r)-i].Category]
		topk[r[len(r)-i].Category] = count + 1
	}
	return topk
}

func (r topResults) query(k int) string {
	max := 0
	var category string
	topk := r.topK(int(math.Min(float64(k), float64(len(r)))))

	for cat, count := range topk {
		if count > max {
			max = count
			category = cat
		}
	}

	return category
}

type topResult struct {
	Score    float64
	Category string
}

func (t *topResult) String() string {
	return fmt.Sprintf("%.2f", t.Score)
}

func asReader(text string) io.Reader {
	return bytes.NewBufferString(text)
}
