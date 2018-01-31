package classifier

// Predicate provides a predicate function
type Predicate func(string) bool

// Mapper provides a map function
type Mapper func(string) string

// Map applies f to each element of the supplied input slice
func Map(vs chan string, f Mapper) chan string {
	outstream := make(chan string)
	go func() {
		for v := range vs {
			outstream <- f(v)
		}
		close(outstream)
	}()
	return outstream
}

// Filter removes elements from the input slice where the supplied predicate
// is satisfied
func Filter(vs chan string, f Predicate) chan string {
	outstream := make(chan string)
	go func() {
		for v := range vs {
			if f(v) {
				outstream <- v
			}
		}
		close(outstream)
	}()
	return outstream
}
