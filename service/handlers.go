package service

import (
	"encoding/json"
	"errors"
	log "github.com/Sirupsen/logrus"
	"github.com/n3integration/classifier"
	"io/ioutil"
	"net/http"
)

// data transfer object
type ClassificationData struct {
	Document string `json:"doc,omitempty"`
	Category string `json:"category,omitempty"`
}

// constructs a new ClassificationData instance from the request body.
// returns an error in the case that the request body is empty or cannot
// be parsed.
func newClassificationData(req *http.Request) (*ClassificationData, error) {
	bytes, err := ioutil.ReadAll(req.Body)
	if err != nil || len(bytes) == 0 {
		return nil, errors.New("request body is required")
	}
	data := ClassificationData{}
	if err := json.Unmarshal(bytes, &data); err != nil {
		return nil, errors.New("malformed request body")
	}
	if data.Document == "" {
		return nil, errors.New("doc is required")
	}
	return &data, nil
}

// classification
type ClassificationHandler struct {
	classifier classifier.Classifier
}

func NewClassificationHandler(classifier classifier.Classifier) *ClassificationHandler {
	return &ClassificationHandler{classifier}
}

// classification request handler
func (this *ClassificationHandler) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	switch req.Method {
	case "POST":
		data, err := newClassificationData(req)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		category, err := this.classifier.Classify(data.Document)
		if err != nil {
			log.Error(err)
			http.Error(w, http.StatusText(http.StatusInternalServerError), http.StatusInternalServerError)
			return
		}
		json, _ := json.Marshal(&ClassificationData{Category: category})
		w.Header().Add("Content-Type", "application/json")
		w.Write([]byte(json))
	default:
		http.Error(w, http.StatusText(http.StatusMethodNotAllowed), http.StatusMethodNotAllowed)
	}
}

// training
type TrainingHandler struct {
	classifier classifier.Classifier
}

func NewTrainingHandler(classifier classifier.Classifier) *TrainingHandler {
	return &TrainingHandler{classifier}
}

// training request handler
func (this *TrainingHandler) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	switch req.Method {
	case "POST":
		data, err := newClassificationData(req)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		err = this.classifier.Train(data.Document, data.Category)
		if err != nil {
			log.Error(err)
			http.Error(w, http.StatusText(http.StatusInternalServerError), http.StatusInternalServerError)
			return
		}
		http.Error(w, "", http.StatusNoContent)
	default:
		http.Error(w, http.StatusText(http.StatusMethodNotAllowed), http.StatusMethodNotAllowed)
	}
}
