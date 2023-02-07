package knn

import (
	"fmt"
	"log"
	"os"
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
		category, err := knn.ClassifyString(data.Headline)
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
	return knn.Train(f, category)
}
