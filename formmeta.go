package dit

import (
	"github.com/PuerkitoBio/goquery"
	"github.com/happyhackingspace/dit/classifier"
	"github.com/happyhackingspace/dit/internal/htmlutil"
)

// FormMeta describes the HTML <form> element carrying a classification.
// It gives RPA tooling stable locator and submission metadata in addition to
// the model's predicted type.
type FormMeta struct {
	// ID is the form's id attribute, empty when absent.
	ID string `json:"id"`
	// Action is the form's action attribute, empty when absent.
	Action string `json:"action"`
	// Method is the form's method lowercased; defaults to "get" when missing.
	Method string `json:"method"`
}

// FieldInfo is one document-ordered, per-control field classification.
//
// Every visible control (input, textarea, select, button except hidden
// inputs) produces one entry, including controls without a name attribute.
// Duplicate names are never folded: each control gets its own entry with a
// unique DOM path, so batch login/registration automation can generate a
// stable RPA action per control.
type FieldInfo struct {
	// Name is the control's name attribute; empty for unnamed controls.
	Name string `json:"name"`
	// ID is the control's id attribute, empty when absent.
	ID string `json:"id"`
	// DOMPath is a stable CSS-selector-like path from the document root,
	// e.g. html:nth-of-type(1) > body:nth-of-type(1) > form:nth-of-type(1) > input:nth-of-type(2).
	DOMPath string `json:"dom_path"`
	// InputType is the HTML control type: the lowercased input type
	// (defaulting to "text") or the tag name for textarea/select/button.
	InputType string `json:"input_type"`
	// Type is the predicted field type label.
	Type string `json:"type"`
	// Confidence is the probability of the predicted Type label (0..1).
	Confidence float64 `json:"confidence"`
}

// FieldInfoProba is the probability variant of FieldInfo. Proba holds the
// per-label probabilities after thresholding; Type/Confidence always reflect
// the unthresholded best label so the two remain populated.
type FieldInfoProba struct {
	Name       string             `json:"name"`
	ID         string             `json:"id"`
	DOMPath    string             `json:"dom_path"`
	InputType  string             `json:"input_type"`
	Type       string             `json:"type"`
	Confidence float64            `json:"confidence"`
	Proba      map[string]float64 `json:"proba,omitempty"`
}

// buildFormMeta extracts locator and submission metadata from a form element.
func buildFormMeta(form *goquery.Selection) FormMeta {
	return FormMeta{
		ID:     htmlutil.GetFormID(form),
		Action: htmlutil.GetFormAction(form),
		Method: htmlutil.GetFormMethodNormalized(form),
	}
}

// toFieldInfos converts classifier-level ordered fields into the hard-label
// public shape.
func toFieldInfos(ordered []classifier.FieldClassification) []FieldInfo {
	if len(ordered) == 0 {
		return nil
	}
	out := make([]FieldInfo, len(ordered))
	for i, f := range ordered {
		out[i] = FieldInfo{
			Name:       f.Name,
			ID:         f.ID,
			DOMPath:    f.DOMPath,
			InputType:  f.InputType,
			Type:       f.Type,
			Confidence: f.Confidence,
		}
	}
	return out
}

// toFieldInfoProbas converts classifier-level ordered fields into the
// probability public shape.
func toFieldInfoProbas(ordered []classifier.FieldClassification) []FieldInfoProba {
	if len(ordered) == 0 {
		return nil
	}
	out := make([]FieldInfoProba, len(ordered))
	for i, f := range ordered {
		out[i] = FieldInfoProba{
			Name:       f.Name,
			ID:         f.ID,
			DOMPath:    f.DOMPath,
			InputType:  f.InputType,
			Type:       f.Type,
			Confidence: f.Confidence,
			Proba:      f.Proba,
		}
	}
	return out
}
