package index

import (
	"fmt"
	"sync"
)

// TermIndex provides a term frequency index
type TermIndex struct {
	index int
	terms map[string]*termRef
	sync.RWMutex
}

// NewTermIndex initializes an empty term frequency index
func NewTermIndex(capacity int) *TermIndex {
	return &TermIndex{
		terms: make(map[string]*termRef, capacity),
	}
}

// Add a term to the index
func (i *TermIndex) Add(t string) int {
	i.Lock()
	defer i.Unlock()
	if _, ok := i.terms[t]; ok {
		i.terms[t].incr()
		return i.terms[t].index
	}
	i.terms[t] = &termRef{
		1,
		i.index,
	}
	i.index++
	return i.terms[t].index
}

// IndexOf returns the index of the provided term, or -1 if not found
func (i *TermIndex) IndexOf(term string) int {
	i.RLock()
	defer i.RUnlock()
	if t, ok := i.terms[term]; ok {
		return t.index
	}
	return -1
}

// Frequency returns the term frequency within the index
func (i *TermIndex) Frequency(term string) float64 {
	i.RLock()
	defer i.RUnlock()
	if t, ok := i.terms[term]; ok {
		return t.freq
	}
	return 0
}

// Count returns the number of terms within the index
func (i *TermIndex) Count() int {
	i.RLock()
	defer i.RUnlock()
	return len(i.terms)
}

func (i *TermIndex) String() string {
	i.RLock()
	defer i.RUnlock()
	return fmt.Sprintf("%v", i.terms)
}

// termRef provides a given term's frequency and ref index
type termRef struct {
	freq  float64
	index int
}

func (t *termRef) incr() float64 {
	t.freq++
	return t.freq
}

func (t *termRef) String() string {
	return fmt.Sprintf("%v", t.freq)
}
