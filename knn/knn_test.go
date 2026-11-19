package knn

import (
	"context"
	"fmt"
	"log"
	"os"
	"regexp"
	"sort"
	"strings"
	"testing"

	"github.com/n3integration/classifier"
)

func TestClassifier(t *testing.T) {
	knn := New(
		K(4),
		Similarity(EuclideanDistance),
		WeightScheme(classifier.TermFrequency),
		Tokenizer(classifier.NewTokenizer(
			classifier.Filters(classifier.IsNotStopWord, classifier.IsWord),
			classifier.SplitFunc(classifier.ScanAlphaWords),
		)),
	)

	dataDir, err := os.ReadDir("testdata")
	if err != nil {
		log.Fatal(err)
	}

	for _, file := range dataDir {
		if file.IsDir() {
			dir := file
			files, rErr := os.ReadDir(fmt.Sprintf("testdata/%s", dir.Name()))
			if rErr != nil {
				log.Fatal(rErr)
			}
			for _, f := range files {
				if lErr := load(knn, dir.Name(), fmt.Sprintf("testdata/%s/%s", dir.Name(), f.Name())); lErr != nil {
					t.Fatal(lErr)
				}
			}
		}
	}

	testdata := []struct {
		Name             string
		Headline         string
		ExpectedCategory string
	}{
		{
			Name:             "Business Headline",
			Headline:         `Small Businesses Keep Hiring as Fed Raises Rates to Cool Economy`,
			ExpectedCategory: "business",
		},
		{
			Name:             "Sports Headline",
			Headline:         `How Eagles can win 2023 Super Bowl: Jalen Hurts, dominant offensive line pave the way for championship run`,
			ExpectedCategory: "sports",
		},
	}

	for _, data := range testdata {
		category, err := knn.ClassifyString(context.Background(), data.Headline)
		if err != nil {
			t.Fatalf("failed to classify %s dataDir: %s", data.Name, err)
		}

		if category != data.ExpectedCategory {
			log.Println(knn.matrix)
			t.Fatalf("incorrectly classified %s; expected %s, but got %s", data.Name, data.ExpectedCategory, category)
		}
	}
}

func load(knn *Classifier, category, filename string) error {
	f, err := os.Open(filename)
	if err != nil {
		return fmt.Errorf("failed to load test data: %w", err)
	}
	defer f.Close()
	return knn.Train(context.Background(), f, category)
}

// TestLeaveOneOutAccuracy holds out each labeled document in turn, trains on the
// rest, and requires a minimum overall accuracy.
func TestLeaveOneOutAccuracy(t *testing.T) {
	type doc struct{ category, path string }
	var docs []doc

	categories, err := os.ReadDir("testdata")
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range categories {
		if !c.IsDir() {
			continue
		}
		files, err := os.ReadDir(fmt.Sprintf("testdata/%s", c.Name()))
		if err != nil {
			t.Fatal(err)
		}
		for _, f := range files {
			docs = append(docs, doc{c.Name(), fmt.Sprintf("testdata/%s/%s", c.Name(), f.Name())})
		}
	}

	correct := 0
	for i, held := range docs {
		knn := New(
			K(3),
			Similarity(CosineSimilarity),
			WeightScheme(classifier.TermFrequency),
			Tokenizer(classifier.NewTokenizer(
				classifier.Filters(classifier.IsNotStopWord, classifier.IsWord),
				classifier.SplitFunc(classifier.ScanAlphaWords),
			)),
		)
		for j, d := range docs {
			if i != j {
				if err := load(knn, d.category, d.path); err != nil {
					t.Fatal(err)
				}
			}
		}
		text, err := os.ReadFile(held.path)
		if err != nil {
			t.Fatal(err)
		}
		got, err := knn.ClassifyString(context.Background(), string(text))
		if err != nil {
			t.Fatalf("failed to classify %s: %v", held.path, err)
		}
		if got == held.category {
			correct++
		} else {
			t.Logf("misclassified %s as %s", held.path, got)
		}
	}

	accuracy := float64(correct) / float64(len(docs))
	t.Logf("leave-one-out accuracy: %d/%d (%.0f%%)", correct, len(docs), accuracy*100)
	if accuracy < 0.8 {
		t.Errorf("accuracy %.2f below threshold 0.80", accuracy)
	}
}

var sentenceEnd = regexp.MustCompile(`[.!?]+\s+`)

// chunk splits text into pieces of n sentences each, dropping a short remainder
func chunk(text string, n int) []string {
	sentences := sentenceEnd.Split(strings.Join(strings.Fields(text), " "), -1)
	var chunks []string
	for i := 0; i+n <= len(sentences); i += n {
		chunks = append(chunks, strings.Join(sentences[i:i+n], ". "))
	}
	return chunks
}

// TestChunkedAccuracy expands the labeled articles into many short documents by
// splitting each into multi-sentence chunks. Each article is held out in turn:
// the classifier trains on the chunks of every other article and is tested on
// the held-out article's chunks. Holding out by article avoids leakage, since
// chunks of one article would otherwise make each other trivially easy to classify.
func TestChunkedAccuracy(t *testing.T) {
	type article struct {
		category string
		chunks   []string
	}
	var articles []article

	categories, err := os.ReadDir("testdata")
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range categories {
		if !c.IsDir() {
			continue
		}
		files, err := os.ReadDir(fmt.Sprintf("testdata/%s", c.Name()))
		if err != nil {
			t.Fatal(err)
		}
		for _, f := range files {
			text, err := os.ReadFile(fmt.Sprintf("testdata/%s/%s", c.Name(), f.Name()))
			if err != nil {
				t.Fatal(err)
			}
			articles = append(articles, article{c.Name(), chunk(string(text), 3)})
		}
	}

	correct, total := 0, 0
	for i, held := range articles {
		knn := New(
			K(3),
			Similarity(CosineSimilarity),
			WeightScheme(classifier.TermFrequency),
			Tokenizer(classifier.NewTokenizer(
				classifier.Filters(classifier.IsNotStopWord, classifier.IsWord),
				classifier.SplitFunc(classifier.ScanAlphaWords),
			)),
		)
		for j, a := range articles {
			if i == j {
				continue
			}
			for _, c := range a.chunks {
				if err := knn.TrainString(context.Background(), c, a.category); err != nil {
					t.Fatal(err)
				}
			}
		}
		for _, c := range held.chunks {
			got, err := knn.ClassifyString(context.Background(), c)
			if err != nil {
				t.Fatalf("failed to classify chunk: %v", err)
			}
			total++
			if got == held.category {
				correct++
			}
		}
	}

	accuracy := float64(correct) / float64(total)
	t.Logf("chunked accuracy: %d/%d (%.0f%%)", correct, total, accuracy*100)
	if accuracy < 0.8 {
		t.Errorf("accuracy %.2f below threshold 0.80", accuracy)
	}
}

func TestQueryTieBreak(t *testing.T) {
	// results are ordered by ascending score, so the highest scores are last
	results := topResults{
		{Score: 0.1, Category: "c"},
		{Score: 0.5, Category: "b"},
		{Score: 0.6, Category: "a"},
		{Score: 0.9, Category: "b"},
		{Score: 0.7, Category: "a"},
	}
	sort.Sort(results)

	for i := 0; i < 50; i++ {
		// k=4 votes a=2, b=2; b wins on higher total similarity (1.4 vs 1.3)
		if got := results.query(4); got != "b" {
			t.Fatalf("expected b, got %s", got)
		}
	}

	equal := topResults{
		{Score: 0.5, Category: "z"},
		{Score: 0.5, Category: "m"},
	}
	for i := 0; i < 50; i++ {
		if got := equal.query(2); got != "m" {
			t.Fatalf("expected m on equal score tie, got %s", got)
		}
	}
}
