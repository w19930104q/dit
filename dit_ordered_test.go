package dit

import (
	"encoding/json"
	"math"
	"reflect"
	"testing"

	"github.com/PuerkitoBio/goquery"
	"github.com/happyhackingspace/dit/classifier"
	"github.com/happyhackingspace/dit/crf"
	"github.com/happyhackingspace/dit/internal/htmlutil"
)

// fixtureHTML contains two forms: a login form with repeated field names, an
// unnamed control and a reCAPTCHA, and a search form.
const fixtureHTML = `
<html><head><title>Sign in</title></head><body>
<form id="loginForm" action="/login" method="POST">
  <input type="text" name="account" id="acct1"/>
  <input type="text" name="account" id="acct2"/>
  <input type="text" id="nameless"/>
  <input type="password" name="password" id="pass"/>
  <input type="hidden" name="csrf" value="token"/>
  <div class="g-recaptcha" data-sitekey="key"></div>
  <button type="submit">Sign in</button>
</form>
<form id="searchForm" action="/search">
  <input type="search" name="q" id="query"/>
  <button type="submit">Search</button>
</form>
</body></html>`

// nestedFormHTML places a <form> inside another form. Per the HTML spec the
// parser drops the inner start tag, but surviving controls must still be
// classified and receive stable DOM paths.
const nestedFormHTML = `
<html><body>
<form id="outer" action="/o" method="post">
  <div><form id="inner" action="/i" method="get"><input name="inside" id="in1"/></form></div>
  <input name="outerfield" id="of1"/>
</form>
</body></html>`

// newTestClassifier trains a tiny but real form/page classifier on two local
// fixtures and plugs in a deterministic hand-built CRF for field labels.
// No model file or network access is required.
func newTestClassifier(t *testing.T) *Classifier {
	t.Helper()

	loginDoc, err := htmlutil.LoadHTMLString(fixtureHTML)
	if err != nil {
		t.Fatal(err)
	}
	plainDoc, err := htmlutil.LoadHTMLString(`<html><body>
<form id="searchForm" action="/search"><input type="text" name="q"/><button>Go</button></form>
</body></html>`)
	if err != nil {
		t.Fatal(err)
	}

	forms := htmlutil.GetForms(loginDoc)
	trainForms := []*goquery.Selection{forms[0], forms[1]}
	formLabels := []string{"login", "search"}
	formModel := classifier.TrainFormType(trainForms, formLabels, classifier.DefaultFormTypeTrainConfig())

	pageDocs := []*goquery.Document{loginDoc, plainDoc}
	pageFormResults := [][]classifier.ClassifyResult{
		{{Form: "login"}},
		{{Form: "search"}},
	}
	urls := []string{"https://example.test/login", "https://example.test/search"}
	pageModel := classifier.TrainPageType(pageDocs, pageFormResults, urls,
		[]string{"login", "search"}, classifier.DefaultPageTypeTrainConfig())

	fieldModel := &classifier.FieldTypeModel{CRF: newDeterministicCRF()}

	return &Classifier{fc: &classifier.FormFieldClassifier{
		FormModel:  formModel,
		FieldModel: fieldModel,
		PageModel:  pageModel,
	}}
}

// newDeterministicCRF builds a two-label CRF where "field" always wins with a
// marginal of sigmoid(3) ≈ 0.953, independently of input features.
func newDeterministicCRF() *crf.Model {
	m := crf.NewModel()
	m.Labels.Add("other") // id 0
	m.Labels.Add("field") // id 1
	m.Attributes.Add("bias")
	m.NumLabels = 2
	m.Weights = make([]float64, m.NumWeights())
	m.Weights[1] = 3.0
	return m
}

// TestExtractFormsOrderedMetadata verifies the extended output keeps legacy
// fields while adding document-ordered metadata, duplicate names and unnamed
// controls, and per-form CAPTCHA detection.
func TestExtractFormsOrderedMetadata(t *testing.T) {
	c := newTestClassifier(t)

	results, err := c.ExtractForms(fixtureHTML)
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 2 {
		t.Fatalf("expected 2 forms, got %d", len(results))
	}

	login := results[0]
	if login.Form.ID != "loginForm" {
		t.Errorf("form id = %q, want loginForm", login.Form.ID)
	}
	if login.Form.Action != "/login" {
		t.Errorf("form action = %q, want /login", login.Form.Action)
	}
	if login.Form.Method != "post" {
		t.Errorf("form method = %q, want post", login.Form.Method)
	}
	if login.Captcha != "recaptcha" {
		t.Errorf("form captcha = %q, want recaptcha", login.Captcha)
	}

	// Legacy name-keyed map is still present and folds the duplicate.
	if login.Fields == nil || login.Fields["account"] != "field" {
		t.Errorf("legacy fields map missing or wrong: %v", login.Fields)
	}
	if _, ok := login.Fields[""]; ok {
		t.Error("legacy map must not contain unnamed controls")
	}

	// Visible controls in form 0: 2x account, nameless, password, button = 5.
	ordered := login.FieldsOrdered
	if len(ordered) != 5 {
		t.Fatalf("expected 5 ordered controls, got %d: %+v", len(ordered), ordered)
	}

	wantNames := []string{"account", "account", "", "password", ""}
	pathSet := make(map[string]bool)
	for i, f := range ordered {
		if f.Name != wantNames[i] {
			t.Errorf("control %d name = %q, want %q", i, f.Name, wantNames[i])
		}
		if f.DOMPath == "" || pathSet[f.DOMPath] {
			t.Fatalf("control %d has empty or duplicate DOM path %q", i, f.DOMPath)
		}
		pathSet[f.DOMPath] = true
		if f.Type != "field" {
			t.Errorf("control %d type = %q, want field", i, f.Type)
		}
		if math.Abs(f.Confidence-0.9525741268224334) > 1e-9 {
			t.Errorf("control %d confidence = %v, want sigmoid(3)", i, f.Confidence)
		}
	}

	// The two account controls must survive as distinct entries.
	if ordered[0].ID != "acct1" || ordered[1].ID != "acct2" {
		t.Errorf("duplicate-name controls not preserved: %q, %q", ordered[0].ID, ordered[1].ID)
	}
	if ordered[0].DOMPath == ordered[1].DOMPath {
		t.Error("duplicate-name controls share a DOM path")
	}
	// Unnamed control is retained with its id and input type.
	if ordered[2].ID != "nameless" || ordered[2].InputType != "text" {
		t.Errorf("unnamed control lost or mistyped: %+v", ordered[2])
	}
	// Button control reports its tag name as input type.
	if ordered[4].InputType != "button" {
		t.Errorf("button input type = %q, want button", ordered[4].InputType)
	}
	// Hidden input is excluded.
	for _, f := range ordered {
		if f.Name == "csrf" {
			t.Error("hidden input must not appear in ordered controls")
		}
	}

	// Second form: missing method defaults to get, no captcha.
	search := results[1]
	if search.Form.ID != "searchForm" || search.Form.Method != "get" {
		t.Errorf("search form metadata wrong: %+v", search.Form)
	}
	if search.Captcha != "" {
		t.Errorf("search form captcha = %q, want empty", search.Captcha)
	}
}

// TestNestedFormHandled covers parser-reconstructed nested-form HTML through
// the full pipeline: one form survives, both controls stay classified, and
// page/standalone entry points agree.
func TestNestedFormHandled(t *testing.T) {
	c := newTestClassifier(t)

	results, err := c.ExtractForms(nestedFormHTML)
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 1 {
		t.Fatalf("nested inner form tag should be dropped, got %d forms", len(results))
	}
	form := results[0]
	if form.Form.ID != "outer" || form.Form.Action != "/o" || form.Form.Method != "post" {
		t.Errorf("outer form metadata wrong: %+v", form.Form)
	}
	// Per the HTML parsing spec the inner form tag is dropped and the outer
	// form closes early, so only the inner control survives inside the form;
	// outerfield is reparented outside any form.
	if len(form.FieldsOrdered) != 1 || form.FieldsOrdered[0].Name != "inside" {
		t.Fatalf("expected only the inner control in the recovered form, got %+v", form.FieldsOrdered)
	}
	if form.FieldsOrdered[0].DOMPath == "" {
		t.Fatal("recovered nested control has empty DOM path")
	}

	page, err := c.ExtractPageType(nestedFormHTML)
	if err != nil {
		t.Fatal(err)
	}
	if len(page.Forms) != 1 || !reflect.DeepEqual(page.Forms[0], results[0]) {
		t.Fatal("page and standalone results differ for nested form HTML")
	}
}

// TestExtractFormsProbaOrdered checks the probability shape: thresholded
// per-label maps, retained best type/confidence, and ordered duplicates.
func TestExtractFormsProbaOrdered(t *testing.T) {
	c := newTestClassifier(t)

	results, err := c.ExtractFormsProba(fixtureHTML, 0.9)
	if err != nil {
		t.Fatal(err)
	}
	login := results[0]

	// Legacy probability map remains keyed by name.
	if login.Fields == nil || login.Fields["account"] == nil {
		t.Fatalf("legacy proba fields missing: %v", login.Fields)
	}
	if _, ok := login.Fields["account"]["other"]; ok {
		t.Error("sub-threshold label should be filtered from legacy proba map")
	}

	ordered := login.FieldsOrdered
	if len(ordered) != 5 {
		t.Fatalf("expected 5 ordered controls, got %d", len(ordered))
	}
	for i, f := range ordered {
		if f.Type != "field" {
			t.Errorf("control %d type = %q, want field", i, f.Type)
		}
		if f.Confidence < 0.9 {
			t.Errorf("control %d confidence = %v, want >= 0.9", i, f.Confidence)
		}
		if f.Proba["field"] != f.Confidence {
			t.Errorf("control %d retained proba %v != confidence %v", i, f.Proba["field"], f.Confidence)
		}
		if _, ok := f.Proba["other"]; ok {
			t.Errorf("control %d sub-threshold label not filtered", i)
		}
	}
}

// TestPageAndStandaloneConsistency is the core parity guarantee: the same
// HTML must produce byte-identical per-form results through the page-level
// and standalone entry points, in both hard and probability modes.
func TestPageAndStandaloneConsistency(t *testing.T) {
	c := newTestClassifier(t)

	standalone, err := c.ExtractForms(fixtureHTML)
	if err != nil {
		t.Fatal(err)
	}
	page, err := c.ExtractPageType(fixtureHTML)
	if err != nil {
		t.Fatal(err)
	}
	if len(page.Forms) != len(standalone) {
		t.Fatalf("form count mismatch: page=%d standalone=%d", len(page.Forms), len(standalone))
	}
	for i := range standalone {
		if !reflect.DeepEqual(page.Forms[i], standalone[i]) {
			t.Fatalf("hard mode form %d differs\npage:      %+v\nstandalone:%+v", i, page.Forms[i], standalone[i])
		}
	}
	if page.Captcha != "recaptcha" {
		t.Errorf("page captcha = %q, want recaptcha", page.Captcha)
	}
	if page.Type == "" {
		t.Error("page type should be populated")
	}

	standaloneP, err := c.ExtractFormsProba(fixtureHTML, 0.05)
	if err != nil {
		t.Fatal(err)
	}
	pageP, err := c.ExtractPageTypeProba(fixtureHTML, 0.05)
	if err != nil {
		t.Fatal(err)
	}
	for i := range standaloneP {
		if !reflect.DeepEqual(pageP.Forms[i], standaloneP[i]) {
			t.Fatalf("proba mode form %d differs\npage:      %+v\nstandalone:%+v", i, pageP.Forms[i], standaloneP[i])
		}
	}
}

// TestOrderedOutputJSONShape verifies the wire format stays backward
// compatible: old keys remain, new keys use the documented names and empty
// values are omitted.
func TestOrderedOutputJSONShape(t *testing.T) {
	c := newTestClassifier(t)
	results, err := c.ExtractForms(fixtureHTML)
	if err != nil {
		t.Fatal(err)
	}

	raw, err := json.Marshal(results[1])
	if err != nil {
		t.Fatal(err)
	}
	var got map[string]any
	if err := json.Unmarshal(raw, &got); err != nil {
		t.Fatal(err)
	}

	for _, key := range []string{"type", "fields", "form", "fields_ordered"} {
		if _, ok := got[key]; !ok {
			t.Errorf("missing JSON key %q in %s", key, raw)
		}
	}
	if _, ok := got["captcha_type"]; ok {
		t.Error("captcha_type must be omitted when no captcha is present")
	}
	formMeta, ok := got["form"].(map[string]any)
	if !ok {
		t.Fatalf("form meta has wrong shape: %T", got["form"])
	}
	if formMeta["method"] != "get" || formMeta["id"] != "searchForm" {
		t.Errorf("form meta JSON wrong: %v", formMeta)
	}
	firstField, ok := got["fields_ordered"].([]any)[0].(map[string]any)
	if !ok {
		t.Fatal("fields_ordered has wrong element shape")
	}
	for _, key := range []string{"name", "id", "dom_path", "input_type", "type", "confidence"} {
		if _, ok := firstField[key]; !ok {
			t.Errorf("ordered field missing JSON key %q", key)
		}
	}
}

// TestMalformedHTMLNoCrash ensures tolerant parsing yields empty or partial
// results without errors or panics, and that recovered controls still get
// metadata.
func TestMalformedHTMLNoCrash(t *testing.T) {
	c := newTestClassifier(t)

	cases := map[string]string{
		"unclosed tags":      `<form action="/x"><input type="text" name="a"><div><input name="b"></form><form><input name="c`,
		"stray text":         `<<<not html>>> formless content`,
		"empty form":         `<form></form>`,
		"form without close": `<form id="oops" action=/y method=post><input name=z id=z1 type=text>`,
	}
	for name, html := range cases {
		t.Run(name, func(t *testing.T) {
			results, err := c.ExtractForms(html)
			if err != nil {
				t.Fatalf("ExtractForms returned error: %v", err)
			}
			for fi, form := range results {
				if form.Form.Method != "get" && form.Form.Method != "post" {
					t.Errorf("form %d method normalized unexpectedly: %q", fi, form.Form.Method)
				}
				for _, f := range form.FieldsOrdered {
					if f.DOMPath == "" {
						t.Errorf("recovered control missing DOM path: %+v", f)
					}
				}
				// Every ordered name must also be traceable in the legacy map
				// when it is non-empty.
				for _, f := range form.FieldsOrdered {
					if f.Name != "" {
						if _, ok := form.Fields[f.Name]; !ok {
							t.Errorf("named ordered control %q absent from legacy map", f.Name)
						}
					}
				}
			}

			// Page entry point must tolerate the same input.
			if _, err := c.ExtractPageType(html); err != nil {
				t.Fatalf("ExtractPageType returned error: %v", err)
			}
		})
	}
}

// TestNoFormsReturnsEmptySlice preserves the original empty-slice contract.
func TestNoFormsReturnsEmptySlice(t *testing.T) {
	c := newTestClassifier(t)
	results, err := c.ExtractForms("<html><body>No forms here</body></html>")
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 0 {
		t.Errorf("expected 0 results, got %d", len(results))
	}
}
