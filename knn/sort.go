package knn

// Partitioning indicates that a type can be partitioned and reordered
type Partitioning interface {
	// Partition between low and high elements
	Partition(low, high int) int
}

func search(values []int, v int) int {
	low := 0
	high := len(values)
	for low <= high {
		mid := (low + high) / 2
		if v == values[mid] {
			return mid
		} else if v > values[mid] {
			low = mid + 1
		} else {
			high = mid - 1
		}
	}
	return -1
}

func quickSort(m Partitioning, low int, high int) {
	stack := make(Stack, 0)

	stack.push(low)
	stack.push(high)
	for stack.len() > 0 {
		high = stack.pop()
		low = stack.pop()

		pivot := m.Partition(low, high)
		if pivot-1 > low {
			stack.push(low)
			stack.push(pivot - 1)
		}

		if pivot+1 < high {
			stack.push(pivot + 1)
			stack.push(high)
		}
	}
}

func swap[V int | float64](a, b *V) {
	t := *a
	*a = *b
	*b = t
}

type Stack []int

func (s *Stack) push(v int) {
	*s = append(*s, v)
}

func (s *Stack) pop() int {
	v := (*s)[len(*s)-1]
	(*s)[len(*s)-1] = 0
	*s = (*s)[:len(*s)-1]
	return v
}

func (s *Stack) len() int {
	return len(*s)
}
