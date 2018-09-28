package classifier

const defaultBufferSize = 50

// Predicate provides a predicate function
type Predicate func(string) bool

// Mapper provides a map function
type Mapper func(string) string

// Map applies f to each element of the supplied input channel
func Map(vs chan string, f ...Mapper) chan string {
	stream := make(chan string, defaultBufferSize)

	go func() {
		for v := range vs {
			for _, fn := range f {
				v = fn(v)
			}
			stream <- v
		}
		close(stream)
	}()

	return stream
}

// Filter removes elements from the input channel where the supplied predicate
// is satisfied
// Filter is a Predicate aggregation
func Filter(vs chan string, filters ...Predicate) chan string {
	stream := make(chan string, defaultBufferSize)
	apply := func(text string) bool {
		for _, f := range filters {
			if !f(text) {
				return false
			}
		}
		return true
	}

	go func() {
		for text := range vs {
			if apply(text) {
				stream <- text
			}
		}
		close(stream)
	}()

	return stream
}