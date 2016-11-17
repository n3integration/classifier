################################################################
# (c) 2016 n3integration
################################################################
.PHONY: clean package test

all: vendor test classifiersvc package

vendor:
	@glide install

classifiersvc: test
	cd cmd/classifiersvc
	CGO_ENABLED=0 GOOS=linux time go build -a -installsuffix cgo -o classifiersvc

package: classifiersvc
	@docker build -t n3integration/classifier .

test: vendor
	@go test -v $(shell glide novendor)

clean:
	cd cmd/classifiersvc && rm -rf classifiersvc
