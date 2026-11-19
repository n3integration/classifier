# classifier
General purpose text classifier (naïve bayes, k-nearest neighbors)

[![codecov](https://codecov.io/gh/n3integration/classifier/branch/master/graph/badge.svg)](https://codecov.io/gh/n3integration/classifier)
[![Go Report Card](https://goreportcard.com/badge/github.com/n3integration/classifier)](https://goreportcard.com/report/github.com/n3integration/classifier)
[![Documentation](https://godoc.org/github.com/n3integration/classifier?status.svg)](http://godoc.org/github.com/n3integration/classifier)

## Installation

```bash
go get github.com/n3integration/classifier
```

## Usage

### Classification

There are two methods of classifying text data: `io.Reader` or `string`. To classify strings, use the `TrainString` 
or `ClassifyString` functions. To classify larger sources, use the `Train` and `Classify` functions that 
take an `io.Reader` as input. All of them accept a `context.Context` as their first argument; cancelling it aborts
the operation and returns the context's error.

```go
package main

import (
	"context"
	"fmt"
	
	"github.com/n3integration/classifier/naive"
)

func main() {
    ctx := context.Background()
    classifier := naive.New()
    classifier.TrainString(ctx, "The quick brown fox jumped over the lazy dog", "ham")
    classifier.TrainString(ctx, "Earn a degree online", "ham")
    classifier.TrainString(ctx, "Earn cash quick online", "spam")

    if classification, err := classifier.ClassifyString(ctx, "Earn your masters degree online"); err == nil {
        fmt.Println("Classification => ", classification) // ham
    } else {
        fmt.Println("error: ", err)
    }	
}
```

## k-NN Encoding

The k-NN classifier encodes each document as a sparse term-weight vector:

1. **Tokenize and count.** The document is tokenized, and occurrences of each term are counted.
2. **Index terms.** A shared `TermIndex` assigns each distinct term an integer ID, which is the term's column in the
   vocabulary.
3. **Weight.** A `WeightScheme` converts each count into a value: `Binary` (presence), `BagOfWords` (raw count),
   `TermFrequency` (`sqrt(count / distinct terms in document)`), or `LogNorm` (`ln(1 + count)`).
4. **Store sparsely.** Each document is a row of (term ID, weight) pairs sorted by term ID, and rows are packed into a
   CSR-style matrix (`ind`, `val`, `ptr` arrays). Only terms present in the document are stored, so memory scales with
   document length rather than vocabulary size.

## Benchmarks

Measured on an Apple M5 Max (darwin/arm64) with `go test -bench=. -benchmem -run=^$ ./...`. The k-NN benchmark
trains on 200 documents across 10 categories (`k=3`); results will vary with hardware and corpus size.

| Package | Benchmark                  | Time/op   | Bytes/op | Allocs/op |
|---------|----------------------------|-----------|----------|-----------|
| `naive` | `BenchmarkClassify`         | 24,345 ns | 12,832 B | 54        |
| `naive` | `BenchmarkClassifyParallel` | 6,343 ns  | 12,834 B | 54        |
| `knn`   | `BenchmarkKNNClassify`         | 14,794 ns | 15,992 B | 232       |
| `knn`   | `BenchmarkKNNClassifyParallel` | 10,494 ns | 15,998 B | 232       |

## Accuracy

The k-NN classifier was evaluated with leave-one-out cross-validation on the 12 labeled news articles in
`knn/testdata` (6 business, 6 sports): each article is held out in turn, the classifier trains on the other 11, and
the held-out article is classified.

| Similarity         | k=1        | k=3        |
|--------------------|------------|------------|
| `EuclideanDistance` | 11/12 (92%) | 8/12 (67%) |
| `CosineSimilarity`  | 11/12 (92%) | 10/12 (83%) |

The corpus is small, allowing a single document to move the result by about 8 points; with that said, treat these 
numbers as a sanity check rather than a general accuracy claim.

## Contributing

- Fork the repository
- Create a local feature branch
- Run `gofmt`
- Bump the `VERSION` file using [semantic versioning](https://semver.org/)
- Submit a pull request

## License

Copyright 2026 n3integration@gmail.com

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
