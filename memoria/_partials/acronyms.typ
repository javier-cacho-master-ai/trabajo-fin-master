#import "table-of-contents.typ": index-heading
#import "header.typ": get-header
#import "styles.typ": styles

// Acronyms read from the shared data file (single source of truth).
#let acronyms-data = yaml("../_acronyms.yml").acronyms

// Lookup by key, for the in-text references.
#let acronyms-by-key = {
  let m = (:)
  for e in acronyms-data { m.insert(e.key, e) }
  m
}

// Tracks which acronyms have already appeared, to distinguish first use.
#let acronyms-seen = state("acronyms-seen", (:))

// In-text reference. First use follows APA 7.ª:
//   long name (SIGLA; traducción)      -- the translation is optional
// subsequent uses show only the sigla. No italics in-text (per APA).
#let acr(key) = {
  let entry = acronyms-by-key.at(key, default: none)
  assert(entry != none, message: "Acrónimo desconocido: '" + key + "'")
  let gloss = if "translation" in entry { "; " + entry.translation } else { "" }
  let full = entry.longname + " (" + entry.shortname + gloss + ")"
  context {
    let first = not acronyms-seen.get().at(key, default: false)
    let shown = if first { full } else { entry.shortname }
    // Link back to the entry in the "Índice de acrónimos".
    link(label("acr-" + key), shown)
  }
  acronyms-seen.update(s => { s.insert(key, true); s })
}

// The list itself: one entry per line, "SIGLA: desarrollo (traducción)", with
// the long name italicised for foreign terms (`italic: true`, the default).
#let acronyms-list() = {
  let sorted = acronyms-data.sorted(key: e => lower(e.shortname))
  for (i, e) in sorted.enumerate() {
    if i > 0 { linebreak() }
    let long = if e.at("italic", default: true) { emph(e.longname) } else { e.longname }
    let gloss = if "translation" in e { " (" + e.translation + ")" } else { "" }
    // The label is the target of the in-text back-links (see `acr`).
    [#strong(e.shortname): #long#gloss#label("acr-" + e.key)]
  }
}

// The "Índice de acrónimos" page, placed in the front matter (after the Índice
// de Tablas), with the same heading style as the other indices.
#let get-acronyms-index(header-text: none) = {
  page(
    header: get-header(header-text: header-text),
    numbering: "I",
    number-align: bottom + right,
  )[
    #set heading(numbering: none)
    #set text(..styles.text-config)
    #set par(..styles.body-paragraph-config)
    #index-heading[Índice de acrónimos]
    #acronyms-list()
  ]
}
