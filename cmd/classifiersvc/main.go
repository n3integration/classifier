package main

import (
	log "github.com/Sirupsen/logrus"
	"github.com/n3integration/classifier/naive"
	"github.com/n3integration/classifier/service"
	"net/http"
	"os"
)

func init() {
	log.SetOutput(os.Stderr)
	log.SetLevel(log.InfoLevel)
}

func main() {
	defaultPort := "9000"

	classifier := naive.NewClassifier()

	http.Handle("/classify", logger(service.NewClassificationHandler(classifier)))
	http.Handle("/train", logger(service.NewTrainingHandler(classifier)))

	log.Info("Listening on ", defaultPort, "...")
	err := http.ListenAndServe(":"+defaultPort, nil)
	if err != nil {
		log.Fatal("ListenAndServe: ", err)
	}
}

// basic request logging middleware
func logger(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		log.Info(r.RemoteAddr, " ", r.Method, " ", r.RequestURI, " ", r.Proto, " ", r.ContentLength)
		next.ServeHTTP(w, r)
	})
}
