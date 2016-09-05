################################################################
# (c) 2016 n3integration
# ################################################################
.PHONY: clean package

all: classifiersvc package

classifiersvc:
	cd cmd/classifiersvc && CGO_ENABLED=0 GOOS=linux time go build -a -installsuffix cgo -o classifiersvc

package: classifiersvc
	docker build -t n3integration/classifier .

clean:
	cd cmd/classifiersvc && rm -rf classifiersvc
