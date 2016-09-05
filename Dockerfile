################################################################
# (c) 2016 n3integration
################################################################
FROM scratch
MAINTAINER "n3integration <n3integration@gmail.com>"
ADD cmd/classifiersvc/classifiersvc /
EXPOSE 9000
CMD ["/classifiersvc"]
