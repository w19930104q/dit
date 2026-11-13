package htmlutil

import (
	"strings"
	"testing"
)

// TestBuildDOMPathUniqueness verifies that every control gets a unique,
// stable path even with duplicate names and missing ids, and that paths are
// anchored at the document root.
func TestBuildDOMPathUniqueness(t *testing.T) {
	html := `
<html><body>
<form id="f1" action="/login" method="post">
  <input type="text" name="token" id="a"/>
  <input type="text" name="token" id="b"/>
  <input type="text" id="noname"/>
  <div><textarea name="note" id="ta"></textarea></div>
</form>
<form id="f2">
  <select name="country" id="c"><option>US</option></select>
  <button type="submit" id="go">Go</button>
</form>
</body></html>`
	doc, err := LoadHTMLString(html)
	if err != nil {
		t.Fatal(err)
	}

	forms := GetForms(doc)
	if len(forms) != 2 {
		t.Fatalf("expected 2 forms, got %d", len(forms))
	}

	fields0 := GetVisibleFields(forms[0])
	if len(fields0) != 4 {
		t.Fatalf("expected 4 visible controls in form 1, got %d", len(fields0))
	}

	pathSet := make(map[string]bool)
	for _, f := range fields0 {
		p := BuildDOMPath(f)
		if !strings.HasPrefix(p, "html:nth-of-type(1) > body:nth-of-type(1) > form:nth-of-type(1)") {
			t.Errorf("path %q not anchored at first form", p)
		}
		if pathSet[p] {
			t.Errorf("duplicate DOM path: %q", p)
		}
		pathSet[p] = true
	}

	// The two same-named inputs must be distinguishable.
	if BuildDOMPath(fields0[0]) == BuildDOMPath(fields0[1]) {
		t.Error("duplicate-name inputs must not share a DOM path")
	}
	// Path must be deterministic across calls.
	if BuildDOMPath(fields0[0]) != BuildDOMPath(fields0[0]) {
		t.Error("DOM path is not stable across calls")
	}
}

// TestBuildDOMPathNthOfType checks that sibling index counts only same-tag
// elements and that deeply nested controls keep their full ancestor chain.
func TestBuildDOMPathNthOfType(t *testing.T) {
	html := `
<html><body>
<form>
  <div><p>x</p><input name="first" id="i1"/></div>
  <div><input name="second" id="i2"/><span>y</span><input name="third" id="i3"/></div>
</form>
</body></html>`
	doc, _ := LoadHTMLString(html)
	form := GetForms(doc)[0]
	fields := GetVisibleFields(form)
	if len(fields) != 3 {
		t.Fatalf("expected 3 inputs, got %d", len(fields))
	}

	third := BuildDOMPath(fields[2])
	if !strings.Contains(third, "div:nth-of-type(2)") {
		t.Errorf("expected second div in path, got %q", third)
	}
	if !strings.HasSuffix(third, "input:nth-of-type(2)") {
		t.Errorf("expected second input within its div, got %q", third)
	}
}

// TestBuildDOMPathNestedForms covers parser-reconstructed nested-form HTML:
// the inner form tag is dropped per HTML spec but surviving controls must
// still receive valid rooted paths.
func TestBuildDOMPathNestedForms(t *testing.T) {
	html := `
<html><body>
<form id="outer" action="/o" method="post">
  <div><form id="inner" action="/i" method="get"><input name="inside" id="in1"/></form></div>
  <input name="outerfield" id="of1"/>
</form>
</body></html>`
	doc, _ := LoadHTMLString(html)
	forms := GetForms(doc)
	if len(forms) != 1 {
		t.Fatalf("parser should drop nested form tag, got %d forms", len(forms))
	}
	// The surviving outer form still contains the inner control.
	fields := GetVisibleFields(forms[0])
	for _, f := range fields {
		p := BuildDOMPath(f)
		if !strings.Contains(p, "form:nth-of-type(") {
			t.Errorf("nested control path %q not rooted at form", p)
		}
	}
}

// TestBuildDOMPathMalformedHTML ensures tolerant parsing never produces empty
// or duplicate paths for recoverable controls.
func TestBuildDOMPathMalformedHTML(t *testing.T) {
	html := `<form action="/broken"><input type="text" name="a"><div><input type="password" name="b"></form>
<form><input name="c><input name=d>`
	doc, err := LoadHTMLString(html)
	if err != nil {
		t.Fatalf("malformed HTML should be tolerated: %v", err)
	}
	forms := GetForms(doc)
	if len(forms) == 0 {
		t.Fatal("expected at least one recovered form")
	}
	globalSeen := make(map[string]bool)
	for _, form := range forms {
		for _, f := range GetVisibleFields(form) {
			p := BuildDOMPath(f)
			if p == "" {
				t.Fatal("empty DOM path for recovered control")
			}
			if globalSeen[p] {
				t.Fatalf("duplicate path across forms: %q", p)
			}
			globalSeen[p] = true
		}
	}
}

// TestFormMetaHelpers checks id/action/method normalization, including the
// GET default and case-insensitive methods.
func TestFormMetaHelpers(t *testing.T) {
	doc, _ := LoadHTMLString(`
<html><body>
<form id="login" action="/auth" method="POST"></form>
<form action="/search"></form>
<form method=" PuT "></form>
</body></html>`)
	forms := GetForms(doc)

	if id := GetFormID(forms[0]); id != "login" {
		t.Errorf("form id = %q, want login", id)
	}
	if action := GetFormAction(forms[0]); action != "/auth" {
		t.Errorf("form action = %q, want /auth", action)
	}
	if method := GetFormMethodNormalized(forms[0]); method != "post" {
		t.Errorf("form method = %q, want post", method)
	}
	if method := GetFormMethodNormalized(forms[1]); method != "get" {
		t.Errorf("missing method = %q, want get", method)
	}
	if method := GetFormMethodNormalized(forms[2]); method != "put" {
		t.Errorf("whitespace method = %q, want put", method)
	}
	if id := GetFormID(forms[1]); id != "" {
		t.Errorf("missing id should be empty, got %q", id)
	}
}

// TestGetFieldType covers input defaults and non-input tag names.
func TestGetFieldType(t *testing.T) {
	doc, _ := LoadHTMLString(`
<form>
  <input name="a"/>
  <input type="EMAIL" name="b"/>
  <input type="checkbox" name="c"/>
  <textarea name="d"></textarea>
  <select name="e"></select>
  <button type="submit" name="f">go</button>
</form>`)
	form := GetForms(doc)[0]
	fields := GetVisibleFields(form)
	want := []string{"text", "email", "checkbox", "textarea", "select", "button"}
	if len(fields) != len(want) {
		t.Fatalf("expected %d controls, got %d", len(want), len(fields))
	}
	for i, f := range fields {
		if got := GetFieldType(f); got != want[i] {
			t.Errorf("control %d type = %q, want %q", i, got, want[i])
		}
	}
}

// TestVisibleFieldsIncludesUnnamed verifies the ordered extraction basis
// keeps unnamed controls while still excluding hidden inputs.
func TestVisibleFieldsIncludesUnnamed(t *testing.T) {
	doc, _ := LoadHTMLString(`
<form>
  <input type="hidden" name="csrf"/>
  <input type="text" name="named"/>
  <input type="text"/>
  <button id="b">x</button>
</form>`)
	form := GetForms(doc)[0]
	visible := GetVisibleFields(form)
	if len(visible) != 3 {
		t.Fatalf("expected 3 visible controls (hidden excluded), got %d", len(visible))
	}
	named := GetFieldsToAnnotate(form)
	if len(named) != 1 {
		t.Fatalf("legacy annotated set should keep only named controls, got %d", len(named))
	}
	if name, _ := named[0].Attr("name"); name != "named" {
		t.Errorf("legacy named control = %q, want named", name)
	}
}
