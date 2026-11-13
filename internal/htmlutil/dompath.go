package htmlutil

import (
	"strconv"
	"strings"

	"github.com/PuerkitoBio/goquery"
	"golang.org/x/net/html"
)

// BuildDOMPath returns a stable, document-order CSS-selector-like path for elem.
//
// The path is anchored at the document root (html) and records, for every
// element on the ancestor chain, its tag name and its 1-based position among
// same-tag siblings (nth-of-type). Positions are always emitted so the path
// stays unique even when the document contains duplicate ids or repeated
// field names, e.g.
// html:nth-of-type(1) > body:nth-of-type(1) > form:nth-of-type(1) > input:nth-of-type(2).
//
// The path depends only on the parsed tree structure, so it is deterministic
// across runs for the same HTML and can be used by RPA tooling to locate a
// specific control.
func BuildDOMPath(elem *goquery.Selection) string {
	if elem == nil || elem.Length() == 0 {
		return ""
	}

	var parts []string
	for node := elem.Get(0); node != nil && node.Type == html.ElementNode; node = node.Parent {
		parts = append(parts, elementPathSegment(node))
	}

	// Walked child -> root; reverse to get root -> child order.
	for i, j := 0, len(parts)-1; i < j; i, j = i+1, j-1 {
		parts[i], parts[j] = parts[j], parts[i]
	}
	return strings.Join(parts, " > ")
}

// elementPathSegment builds the "tag:nth-of-type(k)" segment for a node.
func elementPathSegment(node *html.Node) string {
	position := 1
	for sibling := node.PrevSibling; sibling != nil; sibling = sibling.PrevSibling {
		if sibling.Type == html.ElementNode && sibling.Data == node.Data {
			position++
		}
	}
	return node.Data + ":nth-of-type(" + strconv.Itoa(position) + ")"
}

// GetFormID returns the form element's id attribute (empty when missing).
func GetFormID(form *goquery.Selection) string {
	id, _ := form.Attr("id")
	return id
}

// GetFormMethodNormalized returns the form method lowercased and trimmed.
// HTML forms default to GET when the attribute is absent or blank.
func GetFormMethodNormalized(form *goquery.Selection) string {
	method, _ := form.Attr("method")
	method = strings.ToLower(strings.TrimSpace(method))
	if method == "" {
		return "get"
	}
	return method
}

// GetFieldType reports the control type of a form field.
//
// For <input> elements the type attribute is returned lowercased, defaulting
// to "text" when the attribute is absent or blank, matching the HTML spec.
// For other elements (textarea, select, button) the tag name is returned.
func GetFieldType(elem *goquery.Selection) string {
	tag := goquery.NodeName(elem)
	if tag != "input" {
		return tag
	}
	tp, _ := elem.Attr("type")
	tp = strings.ToLower(strings.TrimSpace(tp))
	if tp == "" {
		return "text"
	}
	return tp
}
