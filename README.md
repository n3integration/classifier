# classifier
A naive bayes text classifier.

## Installation

```bash
go get github.com/n3integration/classifier
```

## Usage

```go
import "github.com/n3integration/classifier/naive"

classifier := naive.NewClassifier()
classifier.Train("The quick brown fox jumped over the lazy dog", "good")
classifier.Train("Earn a degree online", "good")
classifier.Train("Earn cash quick online", "bad")
classification, err := classifier.Classify("Earn your masters degree online")
if err != nil {
    fmt.Println("error: ", err)
} else {
    fmt.Println("Classification => ", classification) // good
}
```
