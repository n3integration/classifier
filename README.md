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

## License

Copyright 2016 n3integration@gmail.com

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
