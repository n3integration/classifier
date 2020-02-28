package knn

import (
	"bytes"
	"fmt"
	"io"
	"math"
	"sort"
	"sync"

	"github.com/n3integration/classifier"
	"github.com/pkg/errors"
)

const (
	defaultKVal          = 1
	defaultIndexCapacity = 1000
)

// Option provides a functional setting for the Classifier
type Option func(c *Classifier) error

// Classifier provides k-nearest neighbor classification
type Classifier struct {
	mu sync.RWMutex

	k            int
	categories   []string
	index        *termIndex
	matrix       *matrix
	similarity   SimilarityScore
	tokenizer    classifier.Tokenizer
	weightScheme WeightSchemeStrategy
}

// New initializes a new k-nearest neighbor classifier unless overridden,
// binary termRef weights and k=1 will be used for the created instance
func New(opts ...Option) *Classifier {
	c := &Classifier{
		k:            defaultKVal,
		categories:   make([]string, 0),
		index:        newIndex(defaultIndexCapacity),
		matrix:       newMatrix(),
		similarity:   CosineSimilarity,
		tokenizer:    classifier.NewTokenizer(),
		weightScheme: Binary,
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

// WeightScheme provides the termRef weight scheme
func WeightScheme(s WeightSchemeStrategy) Option {
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

func (c *Classifier) TrainString(doc string, category string) error {
	return c.Train(asReader(doc), category)
}

func (c *Classifier) Train(r io.Reader, category string) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	wordFreq := make(map[string]float64)
	for text := range c.tokenizer.Tokenize(r) {
		count := wordFreq[text]
		wordFreq[text] = count + 1

		if count == 0 {
			c.index.Add(text)
		}
	}

	c.categories = append(c.categories, category)
	c.matrix.Add(c.index, c.weightScheme(wordFreq), wordFreq)
	return nil
}

func (c *Classifier) ClassifyString(doc string) (string, error) {
	return c.Classify(asReader(doc))
}

func (c *Classifier) Classify(r io.Reader) (string, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	wordFreq := make(map[string]float64)
	for text := range c.tokenizer.Tokenize(r) {
		count := wordFreq[text]
		wordFreq[text] = count + 1
	}

	i := 0
	this := newRow(len(wordFreq))
	for term := range wordFreq {
		this.ind[i] = c.index.IndexOf(term)
		this.val[i] = c.weightScheme(wordFreq)(term)
		i++
	}

	next := c.matrix.Rows()
	results := make(topResults, 0)

	for {
		row := next()
		if row == nil {
			break
		}

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
	topk := r.topK(int(math.Min(float64(k), float64(len(r)))))
	var category string

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

type termIndex struct {
	index int
	terms map[string]*termRef
}

func newIndex(capacity int) *termIndex {
	return &termIndex{
		terms: make(map[string]*termRef, capacity),
	}
}

func (i *termIndex) Add(t string) {
	if _, ok := i.terms[t]; ok {
		i.terms[t].Incr()
		return
	}
	i.terms[t] = &termRef{
		1,
		i.index,
	}
	i.index++
}

func (i *termIndex) IndexOf(term string) int {
	t := i.terms[term]
	if t != nil {
		return t.index
	}
	return -1
}

func (i *termIndex) Frequency(term string) float64 {
	t := i.terms[term]
	if t != nil {
		return t.freq
	}
	return 0
}

func (i *termIndex) Count() int {
	return len(i.terms)
}

func (i *termIndex) String() string {
	return fmt.Sprintf("%v", i.terms)
}

// termRef provides a given termRef's frequency and ref index
type termRef struct {
	freq  float64
	index int
}

func (t *termRef) Incr() float64 {
	t.freq++
	return t.freq
}

func (t *termRef) String() string {
	return fmt.Sprintf("%v", t.freq)
}

// matrix provides compressed row storage
type matrix struct {
	ind []int
	val []float64
	ptr []int
}

func newMatrix() *matrix {
	return &matrix{
		ind: make([]int, 0),
		val: make([]float64, 0),
		ptr: make([]int, 1),
	}
}

func (m *matrix) Add(index *termIndex, weight weightScheme, docWordFreq map[string]float64) {
	for term := range docWordFreq {
		m.ind = append(m.ind, index.IndexOf(term))
		m.val = append(m.val, weight(term))
	}
	last := m.ptr[len(m.ptr)-1]
	m.ptr = append(m.ptr, len(docWordFreq)+last)
}

func (m *matrix) Rows() func() *row {
	i := 0
	r := &row{}

	return func() *row {
		if i == (len(m.ptr) - 1) {
			return nil
		}

		start := m.ptr[i]
		end := m.ptr[i+1]

		r.index = i
		r.ind = m.ind[start:end]
		r.val = m.val[start:end]
		i++

		return r
	}
}

func (m *matrix) Head() []*row {
	iterator := m.Rows()
	count := int(math.Min(10, m.Size()))
	rows := make([]*row, count)

	for i := 0; i <= count; i++ {
		row := iterator()
		if row == nil {
			break
		}
		rows[i] = row
	}

	return rows
}

func (m *matrix) Shape() string {
	return fmt.Sprintf("%v x %v", len(m.ind), len(m.ptr)-1)
}

func (m *matrix) Size() float64 {
	return float64(len(m.ptr)) - 1
}

func (m *matrix) String() string {
	return fmt.Sprintf("%v\n%v\n%v", m.ind, m.val, m.ptr)
}

type row struct {
	ind   []int
	val   []float64
	index int
}

func newRow(size int) *row {
	return &row{
		ind: make([]int, size),
		val: make([]float64, size),
	}
}

func (r *row) Column(i int) (int, float64) {
	return r.ind[i], r.val[i]
}

func (r *row) Feature(i int) int {
	return r.ind[i]
}

func (r *row) Sum() float64 {
	sum := 0.0
	for _, val := range r.val {
		sum += val
	}
	return sum
}

func (r *row) Square() float64 {
	sum := 0.0
	for _, val := range r.val {
		sum += math.Pow(val, 2)
	}
	return sum
}

func (r *row) L2Norm() float64 {
	return math.Sqrt(r.Square())
}

func (r *row) Dot(other *row) float64 {
	sum := 0.0
	if r.Size() <= other.Size() {
		for i := 0; i < len(r.ind); i++ {
			feature, val := r.Column(i)
			sum += val * other.Value(feature)
		}
	} else {
		for i := 0; i < len(other.ind); i++ {
			feature, val := other.Column(i)
			sum += val * r.Value(feature)
		}
	}
	return sum
}

func (r *row) Value(feature int) float64 {
	for i := 0; i < len(r.ind); i++ {
		if r.ind[i] == feature {
			return r.val[i]
		}
	}
	return 0
}

func (r *row) Values(features ...int) *row {
	other := newRow(len(features))
	for i := 0; i < len(features); i++ {
		other.ind[i] = features[i]
		other.val[i] = r.Value(features[i])
	}
	return other
}

func (r *row) Contains(feature int) bool {
	for _, val := range r.ind {
		if val == feature {
			return true
		}
	}
	return false
}

func (r *row) Index() int {
	return r.index
}

func (r *row) Len() int {
	return len(r.ind)
}

func (r *row) Less(i, j int) bool {
	return r.ind[i] < r.ind[j]
}

func (r *row) Swap(i, j int) {
	ind := r.ind[i]
	r.ind[i] = r.ind[j]
	r.ind[j] = ind

	val := r.val[i]
	r.val[i] = r.val[j]
	r.val[j] = val
}

func (r *row) Size() float64 {
	return float64(len(r.val))
}

func (r *row) String() string {
	return fmt.Sprintf("%v\n%v", r.ind, r.val)
}

func asReader(text string) io.Reader {
	return bytes.NewBufferString(text)
}
