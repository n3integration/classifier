package classifier

// Predicate provides a predicate function
type Predicate func(string) bool

// Mapper provides a map function
type Mapper func(string) string

// Map applies f to each element of the supplied input slice
func Map(vs []string, f Mapper) []string {
	vsm := make([]string, len(vs))
	for i, v := range vs {
		vsm[i] = f(v)
	}
	return vsm
}

// Filter removes elements from the input slice where the supplied predicate
// is satisfied
func Filter(vs []string, f Predicate) []string {
	vsf := make([]string, 0)
	for _, v := range vs {
		if f(v) {
			vsf = append(vsf, v)
		}
	}
	return vsf
}
