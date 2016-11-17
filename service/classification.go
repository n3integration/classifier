package service

import (
	"encoding/json"
	"net/http"

	log "github.com/Sirupsen/logrus"

	"github.com/n3integration/classifier"
)

// ClassificationHandler services document classification requests
type ClassificationHandler struct {
	classifier classifier.Classifier
}

// NewClassificationHandler initializes a new ClassificationHandler
func NewClassificationHandler(classifier classifier.Classifier) *ClassificationHandler {
	return &ClassificationHandler{classifier}
}

func (h *ClassificationHandler) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	if req.Method != http.MethodPost {
		http.Error(w, http.StatusText(http.StatusMethodNotAllowed), http.StatusMethodNotAllowed)
		return
	}

	data, err := newClassificationData(req)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	category, err := h.classifier.Classify(data.Document)
	if err != nil {
		log.Error("failed to classify document", err)
		http.Error(w, http.StatusText(http.StatusInternalServerError), http.StatusInternalServerError)
		return
	}

	var response = &ClassificationData{Category: category}
	if json, err := json.Marshal(response); err != nil {
		log.Error("failed to marshal response", err)
		http.Error(w, http.StatusText(http.StatusInternalServerError), http.StatusInternalServerError)
	} else {
		w.Write([]byte(json))
	}
}
