# classifier
A naive bayes text classifier

## Usage

```go
naive := NewNaiveClassifier()
naive.Train("The quick brown fox jumped over the lazy dog", "good")
naive.Train("Earn a degree online", "good")
naive.Train("Earn cash quick online", "bad")
classification, err := naive.Classify("Earn your masters degree online")
if err != nil {
    fmt.Println("error: ", err)
} else {
    fmt.Println("Classification => ", classification) // good
}
```
