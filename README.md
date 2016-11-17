# classifier
A naive bayes text classifier.

## Installation

```bash
go get github.com/n3integration/classifier
```

## Usage

```go
import "github.com/n3integration/classifier/naive"

classifier := naive.New()
classifier.Train("The quick brown fox jumped over the lazy dog", "ham")
classifier.Train("Earn a degree online", "ham")
classifier.Train("Earn cash quick online", "spam")

if classification, err := classifier.Classify("Earn your masters degree online"); err == nil {
    fmt.Println("Classification => ", classification) // ham
} else {
    fmt.Println("error: ", err)
}
```
