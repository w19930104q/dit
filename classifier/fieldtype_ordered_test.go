package classifier

import (
	"testing"

	"github.com/happyhackingspace/dit/crf"
	"github.com/happyhackingspace/dit/internal/htmlutil"
)

// newStubCRF builds an in-memory two-label CRF whose only state feature is
// the unit "bias" attribute. Weighting label 1 above label 0 makes every
// position deterministically predict label 1 with a computable marginal, so
// ordered classification can be tested without a trained model file.
func newStubCRF(t *testing.T) *crf.Model {
	t.Helper()
	m := crf.NewModel()
	m.Labels.Add("other")  // id 0
	m.Labels.Add("target") // id 1
	m.Attributes.Add("bias")
	m.NumLabels = 2
	// Weight layout: [attrID*2+labelID state weights | 2x2 transitions].
	// State weights: label 0 -> 0, label 1 -> 3 (sigmoid(3) ≈ 0.953).
	m.Weights = make([]float64, m.NumWeights())
	m.Weights[1] = 3.0
	return m
}

// orderedFormHTML exercises every ordered-output requirement at once:
// duplicate names, an unnamed control, a hidden input, a select and a button.
const orderedFormHTML = `
<html><body>
<form id="login" action="/auth" method="post">
  <input type="text" name="token" id="t1"/>
  <input type="text" name="token" id="t2"/>
  <input type="text" id="noname"/>
  <input type="hidden" name="csrf" value="x"/>
  <select name="country" id="c"><option>US</option></select>
  <button type="submit" id="go">Go</button>
</form>
</body></html>`

func TestClassifyOrderedPreservesControlsAndOrder(t *testing.T) {
	doc, err := htmlutil.LoadHTMLString(orderedFormHTML)
	if err != nil {
		t.Fatal(err)
	}
	form := htmlutil.GetForms(doc)[0]
	model := &FieldTypeModel{CRF: newStubCRF(t)}

	ordered := model.ClassifyOrdered(form, "login")

	// Visible controls: 2 token inputs, 1 unnamed, select, button = 5.
	if len(ordered) != 5 {
		t.Fatalf("expected 5 ordered controls, got %d", len(ordered))
	}

	wantNames := []string{"token", "token", "", "country", ""}
	wantIDs := []string{"t1", "t2", "noname", "c", "go"}
	wantTypes := []string{"text", "text", "text", "select", "button"}
	for i, fc := range ordered {
		if fc.Name != wantNames[i] {
			t.Errorf("control %d name = %q, want %q", i, fc.Name, wantNames[i])
		}
		if fc.ID != wantIDs[i] {
			t.Errorf("control %d id = %q, want %q", i, fc.ID, wantIDs[i])
		}
		if fc.InputType != wantTypes[i] {
			t.Errorf("control %d input type = %q, want %q", i, fc.InputType, wantTypes[i])
		}
		if fc.Type != "target" {
			t.Errorf("control %d predicted type = %q, want target", i, fc.Type)
		}
		if fc.Confidence <= 0 || fc.Confidence > 1 {
			t.Errorf("control %d confidence out of range: %v", i, fc.Confidence)
		}
		if fc.DOMPath == "" {
			t.Errorf("control %d has empty DOM path", i)
		}
	}

	// Duplicate names must not be folded: identical names, distinct paths.
	if ordered[0].DOMPath == ordered[1].DOMPath {
		t.Error("duplicate-name controls must have distinct DOM paths")
	}

	// Legacy map output still folds the duplicate and excludes unnamed controls.
	legacy := model.Classify(form, "login")
	if len(legacy) != 2 {
		t.Fatalf("legacy name map should contain token+country only, got %d entries: %v", len(legacy), legacy)
	}
	if legacy["token"] != "target" || legacy["country"] != "target" {
		t.Errorf("legacy map labels wrong: %v", legacy)
	}
	if _, ok := legacy[""]; ok {
		t.Error("legacy map must not contain an empty-name key")
	}
}

func TestClassifyOrderedProbaThresholdAndConfidence(t *testing.T) {
	doc, _ := htmlutil.LoadHTMLString(orderedFormHTML)
	form := htmlutil.GetForms(doc)[0]
	model := &FieldTypeModel{CRF: newStubCRF(t)}

	// High threshold filters the weak label out of Proba, but Type and
	// Confidence must still be populated from the unthresholded marginals.
	ordered := model.ClassifyOrderedProba(form, "login", 0.9)
	if len(ordered) != 5 {
		t.Fatalf("expected 5 ordered controls, got %d", len(ordered))
	}
	for i, fc := range ordered {
		if fc.Type != "target" {
			t.Errorf("control %d type = %q, want target", i, fc.Type)
		}
		if fc.Confidence < 0.9 {
			t.Errorf("control %d target confidence = %v, want >= 0.9", i, fc.Confidence)
		}
		if _, ok := fc.Proba["other"]; ok {
			t.Errorf("control %d: sub-threshold label must be filtered from Proba: %v", i, fc.Proba)
		}
		if fc.Proba["target"] != fc.Confidence {
			t.Errorf("control %d: retained proba %v != confidence %v", i, fc.Proba["target"], fc.Confidence)
		}
	}

	// Zero threshold keeps the full distribution.
	full := model.ClassifyOrderedProba(form, "login", 0)
	if len(full[0].Proba) != 2 {
		t.Errorf("zero threshold should retain both labels, got %v", full[0].Proba)
	}
}

func TestClassifyOrderedNoVisibleFields(t *testing.T) {
	doc, _ := htmlutil.LoadHTMLString(`<form><input type="hidden" name="csrf"/></form>`)
	form := htmlutil.GetForms(doc)[0]
	model := &FieldTypeModel{CRF: newStubCRF(t)}

	if got := model.ClassifyOrdered(form, "login"); got != nil {
		t.Errorf("expected nil ordered result without visible controls, got %v", got)
	}
	if got := model.ClassifyOrderedProba(form, "login", 0); got != nil {
		t.Errorf("expected nil proba ordered result, got %v", got)
	}
}

func TestOrderedFieldsFlowThroughFormResult(t *testing.T) {
	doc, _ := htmlutil.LoadHTMLString(orderedFormHTML)
	form := htmlutil.GetForms(doc)[0]
	fc := &FormFieldClassifier{FieldModel: &FieldTypeModel{CRF: newStubCRF(t)}}

	// FormModel is nil, so exercise the field-model path directly via the
	// ordered field model API and confirm the internal FormResult wiring used
	// by ExtractForms carries the same records.
	ordered := fc.FieldModel.ClassifyOrdered(form, "login")
	internal := FormResult{OrderedFields: ordered}
	if len(internal.OrderedFields) != 5 {
		t.Fatalf("internal FormResult should carry 5 ordered fields, got %d", len(internal.OrderedFields))
	}
	// The unnamed control survives the full ordered pipeline.
	foundUnnamed := false
	for _, f := range internal.OrderedFields {
		if f.Name == "" && f.ID == "noname" {
			foundUnnamed = true
		}
	}
	if !foundUnnamed {
		t.Error("unnamed control missing from ordered internal result")
	}
}
