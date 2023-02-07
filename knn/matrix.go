package knn

import (
	"fmt"
	"math"

	"github.com/n3integration/classifier"
	"github.com/n3integration/classifier/index"
)

// sparse matrix implementation
type sparse struct {
	ind []int
	val []float64
	ptr []int
}

// newSparseMatrix initializes an empty sparse matrix
func newSparseMatrix() *sparse {
	return &sparse{
		ind: make([]int, 0),
		val: make([]float64, 0),
		ptr: make([]int, 1),
	}
}

// Add a new row to the underlying matrix
func (m *sparse) Add(index *index.TermIndex, weight classifier.WeightScheme, docWordFreq map[string]float64) {
	prev := len(m.ind)
	for term := range docWordFreq {
		m.ind = append(m.ind, index.IndexOf(term))
		m.val = append(m.val, weight(term))
	}

	cur := prev + len(docWordFreq)
	quickSort(m, prev, cur-1)
	m.ptr = append(m.ptr, cur)
}

// MakeRow creates and returns a new sparseRow without adding it to the underlying matrix
func (m *sparse) MakeRow(index *index.TermIndex, weight classifier.WeightSchemeStrategy, wordFreq map[string]float64) *sparseRow {
	i := 0
	var idx int
	this := newSparseRow(len(wordFreq))

	for term := range wordFreq {
		idx = index.IndexOf(term)
		if idx < 0 {
			idx = index.Add(term)
		}
		this.ind[i] = idx
		this.val[i] = weight(wordFreq)(term)
		i++
	}

	quickSort(this, 0, len(wordFreq)-1)
	return this
}

// Rows returns an iterator over the matrix
func (m *sparse) Rows() func() *sparseRow {
	i := 0
	r := &sparseRow{}

	return func() *sparseRow {
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

// Head returns the first 10 rows in the underlying matrix
func (m *sparse) Head() []*sparseRow {
	iterator := m.Rows()
	count := int(math.Min(10, m.Size()))
	rows := make([]*sparseRow, count)

	for i := 0; i <= count; i++ {
		row := iterator()
		if row == nil {
			break
		}
		rows[i] = row
	}

	return rows
}

func (m *sparse) Shape() string {
	return fmt.Sprintf("%v x %v", len(m.ind), len(m.ptr)-1)
}

func (m *sparse) Size() float64 {
	return float64(len(m.ptr)) - 1
}

func (m *sparse) Partition(low int, high int) int {
	x := m.ind[high]
	i := low - 1

	for j := low; j <= high-1; j++ {
		if m.ind[j] <= x {
			i++
			swap(&m.ind[i], &m.ind[j])
			swap(&m.val[i], &m.val[j])
		}
	}
	swap(&m.ind[i+1], &m.ind[high])
	swap(&m.val[i+1], &m.val[high])
	return i + 1
}

func (m *sparse) String() string {
	return fmt.Sprintf("%v\n%v\n%v", m.ind, m.val, m.ptr)
}

type sparseRow struct {
	ind   []int
	val   []float64
	index int
}

func newSparseRow(size int) *sparseRow {
	return &sparseRow{
		ind: make([]int, size),
		val: make([]float64, size),
	}
}

// Column returns the feature and value at index i
func (r *sparseRow) Column(i int) (int, float64) {
	return r.ind[i], r.val[i]
}

// Feature returns the feature at index i
func (r *sparseRow) Feature(i int) int {
	return r.ind[i]
}

// Sum the sparseRow
func (r *sparseRow) Sum() float64 {
	sum := 0.0
	for _, val := range r.val {
		sum += val
	}
	return sum
}

// Square the row
func (r *sparseRow) Square() float64 {
	sum := 0.0
	for _, val := range r.val {
		sum += math.Pow(val, 2)
	}
	return sum
}

// L2Norm returns the euclidean distance
func (r *sparseRow) L2Norm() float64 {
	return math.Sqrt(r.Square())
}

// Dot returns the dot product
func (r *sparseRow) Dot(other *sparseRow) float64 {
	sum := 0.0
	if r.Size() <= other.Size() {
		for i := 0; i < r.Len(); i++ {
			feature, val := r.Column(i)
			sum += val * other.Value(feature)
		}
	} else {
		for i := 0; i < other.Len(); i++ {
			feature, val := other.Column(i)
			sum += val * r.Value(feature)
		}
	}
	return sum
}

// Value returns the value of feature
func (r *sparseRow) Value(feature int) float64 {
	i := search(r.ind, feature)
	if i >= 0 {
		return r.val[i]
	}
	return 0
}

// Values constructs a new sparse row from the provided features
func (r *sparseRow) Values(features ...int) *sparseRow {
	other := newSparseRow(len(features))
	for i := 0; i < len(features); i++ {
		other.ind[i] = features[i]
		other.val[i] = r.Value(features[i])
	}
	return other
}

// Contains to check if row contains the provided feature
func (r *sparseRow) Contains(feature int) bool {
	for _, val := range r.ind {
		if val == feature {
			return true
		}
	}
	return false
}

// Index returns the index pointer
func (r *sparseRow) Index() int {
	return r.index
}

// Len returns the number of columns
func (r *sparseRow) Len() int {
	return len(r.ind)
}

func (r *sparseRow) Less(i, j int) bool {
	return r.ind[i] < r.ind[j]
}

func (r *sparseRow) Swap(i, j int) {
	ind := r.ind[i]
	r.ind[i] = r.ind[j]
	r.ind[j] = ind

	val := r.val[i]
	r.val[i] = r.val[j]
	r.val[j] = val
}

func (r *sparseRow) Size() float64 {
	return float64(len(r.val))
}

func (r *sparseRow) Partition(low int, high int) int {
	x := r.ind[high]
	i := low - 1

	for j := low; j <= high-1; j++ {
		if r.ind[j] <= x {
			i++
			swap(&r.ind[i], &r.ind[j])
			swap(&r.val[i], &r.val[j])
		}
	}
	swap(&r.ind[i+1], &r.ind[high])
	swap(&r.val[i+1], &r.val[high])
	return i + 1
}

func (r *sparseRow) String() string {
	return fmt.Sprintf("%v\n%v", r.ind, r.val)
}
