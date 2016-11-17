// service provides the web service handlers
package service

import (
	"encoding/json"
	"errors"
	"io/ioutil"
	"net/http"
)

// ClassificationData is a basic data transfer object
type ClassificationData struct {
	Document string `json:"doc,omitempty"`
	Category string `json:"category,omitempty"`
}

// newClassificationData constructs a new ClassificationData instance from
// the request body. Returns an error in the case that the request body is
// empty or cannot be parsed.
func newClassificationData(req *http.Request) (*ClassificationData, error) {
	bytes, err := ioutil.ReadAll(req.Body)
	if err != nil || len(bytes) == 0 {
		return nil, errors.New("request body is required")
	}
	data := &ClassificationData{}
	if err := json.Unmarshal(bytes, data); err != nil {
		return nil, errors.New("malformed request body")
	}
	if data.Document == "" {
		return nil, errors.New("doc is required")
	}
	return data, nil
}
