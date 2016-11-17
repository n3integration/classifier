package service

import (
	"net/http"

	log "github.com/Sirupsen/logrus"

	"github.com/n3integration/classifier"
)

// TrainingHandler services supervised classifier training requests
type TrainingHandler struct {
	classifier classifier.Classifier
}

// NewTrainingHandler initializes a new TrainingHandler
func NewTrainingHandler(classifier classifier.Classifier) *TrainingHandler {
	return &TrainingHandler{classifier}
}

func (h *TrainingHandler) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	if req.Method != http.MethodPost {
		http.Error(w, http.StatusText(http.StatusMethodNotAllowed), http.StatusMethodNotAllowed)
		return
	}

	data, err := newClassificationData(req)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	if err = h.classifier.Train(data.Document, data.Category); err != nil {
		log.Error(err)
		http.Error(w, http.StatusText(http.StatusInternalServerError), http.StatusInternalServerError)
		return
	}

	w.WriteHeader(http.StatusNoContent)
}
