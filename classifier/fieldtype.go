package classifier

import (
	"github.com/PuerkitoBio/goquery"
	"github.com/happyhackingspace/dit/crf"
	"github.com/happyhackingspace/dit/internal/htmlutil"
)

// FieldTypeModel wraps a CRF model for field type classification.
type FieldTypeModel struct {
	CRF *crf.Model
}

// FieldClassification holds the ordered classification of a single form
// control. Unlike the name-keyed maps returned by Classify/ClassifyProba,
// every visible control gets one entry — including controls without a name
// attribute — and entries are kept in document order, so duplicate names are
// never collapsed.
type FieldClassification struct {
	Name       string             `json:"name"`
	ID         string             `json:"id"`
	DOMPath    string             `json:"dom_path"`
	InputType  string             `json:"input_type"`
	Type       string             `json:"type"`
	Confidence float64            `json:"confidence"`
	Proba      map[string]float64 `json:"proba,omitempty"`
}

// Classify returns field types keyed by field name.
//
// For backward compatibility this only covers visible controls carrying a
// non-empty name attribute; repeated names overwrite each other in the map.
// Callers needing stable per-control results should use ClassifyOrdered.
func (m *FieldTypeModel) Classify(form *goquery.Selection, formType string) map[string]string {
	fieldElems := htmlutil.GetFieldsToAnnotate(form)
	if len(fieldElems) == 0 {
		return nil
	}

	rawFeatures := GetFormFeatures(form, formType, fieldElems)

	// Convert to CRF attributes
	crfFeatures := make([]map[string]float64, len(rawFeatures))
	for i, feat := range rawFeatures {
		crfFeatures[i] = crf.FeaturesToAttributes(feat)
	}

	// Predict
	labels := m.CRF.Predict(crfFeatures)

	// Map labels back to field names
	result := make(map[string]string, len(fieldElems))
	for i, elem := range fieldElems {
		name, _ := elem.Attr("name")
		if i < len(labels) {
			result[name] = labels[i]
		}
	}
	return result
}

// ClassifyOrdered classifies every visible control of a form in document
// order, including controls without a name attribute. The predicted label is
// the Viterbi path and Confidence is the marginal probability of that label,
// so hard-label callers get a confidence score without changing the model.
func (m *FieldTypeModel) ClassifyOrdered(form *goquery.Selection, formType string) []FieldClassification {
	fieldElems := htmlutil.GetVisibleFields(form)
	if len(fieldElems) == 0 {
		return nil
	}

	rawFeatures := GetFormFeatures(form, formType, fieldElems)
	crfFeatures := make([]map[string]float64, len(rawFeatures))
	for i, feat := range rawFeatures {
		crfFeatures[i] = crf.FeaturesToAttributes(feat)
	}

	labels := m.CRF.Predict(crfFeatures)
	marginals := m.CRF.PredictMarginals(crfFeatures)

	return buildFieldClassifications(fieldElems, labels, marginals, nil, false)
}

// ClassifyProba returns field type probabilities keyed by field name.
//
// Like Classify, this covers only named visible controls and folds duplicate
// names; use ClassifyOrderedProba for name-preserving ordered results.
func (m *FieldTypeModel) ClassifyProba(form *goquery.Selection, formType string) map[string]map[string]float64 {
	fieldElems := htmlutil.GetFieldsToAnnotate(form)
	if len(fieldElems) == 0 {
		return nil
	}

	rawFeatures := GetFormFeatures(form, formType, fieldElems)

	crfFeatures := make([]map[string]float64, len(rawFeatures))
	for i, feat := range rawFeatures {
		crfFeatures[i] = crf.FeaturesToAttributes(feat)
	}

	marginals := m.CRF.PredictMarginals(crfFeatures)

	result := make(map[string]map[string]float64, len(fieldElems))
	for i, elem := range fieldElems {
		name, _ := elem.Attr("name")
		if i < len(marginals) {
			result[name] = marginals[i]
		}
	}
	return result
}

// ClassifyOrderedProba classifies every visible control in document order and
// attaches the full per-label probability distribution. Type is the highest
// probability label and Confidence is its probability. The returned Proba
// maps are thresholded identically to the legacy name-keyed output.
func (m *FieldTypeModel) ClassifyOrderedProba(form *goquery.Selection, formType string, threshold float64) []FieldClassification {
	fieldElems := htmlutil.GetVisibleFields(form)
	if len(fieldElems) == 0 {
		return nil
	}

	rawFeatures := GetFormFeatures(form, formType, fieldElems)
	crfFeatures := make([]map[string]float64, len(rawFeatures))
	for i, feat := range rawFeatures {
		crfFeatures[i] = crf.FeaturesToAttributes(feat)
	}

	marginals := m.CRF.PredictMarginals(crfFeatures)

	return buildFieldClassifications(fieldElems, nil, marginals, func(probs map[string]float64) map[string]float64 {
		return thresholdMap(probs, threshold)
	}, true)
}

// buildFieldClassifications maps per-element CRF outputs onto ordered
// FieldClassification records. When labels is non-empty (hard mode) the
// Viterbi label is used and confidence is read from its marginal; otherwise
// (proba mode) the best marginal determines both type and confidence.
func buildFieldClassifications(fieldElems []*goquery.Selection, labels []string, marginals []map[string]float64,
	filter func(map[string]float64) map[string]float64, probaMode bool) []FieldClassification {

	out := make([]FieldClassification, len(fieldElems))
	for i, elem := range fieldElems {
		name, _ := elem.Attr("name")
		id, _ := elem.Attr("id")
		fc := FieldClassification{
			Name:      name,
			ID:        id,
			DOMPath:   htmlutil.BuildDOMPath(elem),
			InputType: htmlutil.GetFieldType(elem),
		}

		var marginal map[string]float64
		if i < len(marginals) {
			marginal = marginals[i]
		}

		if !probaMode {
			if i < len(labels) {
				fc.Type = labels[i]
			}
			if marginal != nil {
				fc.Confidence = marginal[fc.Type]
			}
		} else {
			bestLabel := ""
			bestProb := -1.0
			for label, prob := range marginal {
				if prob > bestProb {
					bestProb = prob
					bestLabel = label
				}
			}
			fc.Type = bestLabel
			if bestProb >= 0 {
				fc.Confidence = bestProb
			}
			if marginal == nil {
				fc.Proba = nil
			} else if filter != nil {
				fc.Proba = filter(marginal)
			} else {
				fc.Proba = marginal
			}
		}

		out[i] = fc
	}
	return out
}

// TrainFieldType trains a CRF model for field type classification.
func TrainFieldType(sequences []crf.TrainingSequence, config crf.TrainerConfig) *FieldTypeModel {
	crfModel := crf.Train(sequences, config)
	return &FieldTypeModel{CRF: crfModel}
}
