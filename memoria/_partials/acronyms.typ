#import "table-of-contents.typ": index-heading
#import "header.typ": get-header
#import "styles.typ": styles

// Acrónimos leídos del fichero de datos compartido (fuente única de verdad).
#let acronyms-data = yaml("../_acronyms.yml").acronyms

// Búsqueda por clave, para las menciones en el texto.
#let acronyms-by-key = {
  let m = (:)
  for e in acronyms-data { m.insert(e.key, e) }
  m
}

// Registra qué acrónimos ya han aparecido, para distinguir la primera mención.
#let acronyms-seen = state("acronyms-seen", (:))

// Mención en el texto. La primera encabeza con el término en español:
//   término en español (SIGLA; término original)
// Sin `translation` solo se dispone del desarrollo original: «término (SIGLA)».
// Las siguientes menciones muestran solo la sigla. Sin cursiva en el texto (APA).
#let acr(key, short: false) = {
  let entry = acronyms-by-key.at(key, default: none)
  assert(entry != none, message: "Acrónimo desconocido: '" + key + "'")
  // Enlace de vuelta a la entrada del "Índice de acrónimos".
  let sigla = link(label("acr-" + key), entry.shortname)
  // Encabezados y pies de figura se reproducen en los índices preliminares, así
  // que solo llevan la sigla y no consumen la primera aparición del cuerpo.
  if short { return sigla }

  let full = if "translation" in entry {
    entry.translation + " (" + entry.shortname + "; " + entry.longname + ")"
  } else {
    entry.longname + " (" + entry.shortname + ")"
  }
  context {
    let first = not acronyms-seen.get().at(key, default: false)
    if first { link(label("acr-" + key), full) } else { sigla }
  }
  acronyms-seen.update(s => { s.insert(key, true); s })
}

// El índice: una entrada por línea, «SIGLA: desarrollo (traducción)», con el
// desarrollo en cursiva para los términos extranjeros (`italic: true`, por defecto).
#let acronyms-list() = {
  let sorted = acronyms-data.sorted(key: e => lower(e.shortname))
  for (i, e) in sorted.enumerate() {
    if i > 0 { linebreak() }
    let long = if e.at("italic", default: true) { emph(e.longname) } else { e.longname }
    let gloss = if "translation" in e { " (" + e.translation + ")" } else { "" }
    // La etiqueta es el destino de los enlaces de vuelta del texto (véase `acr`).
    [#strong(e.shortname): #long#gloss#label("acr-" + e.key)]
  }
}

// La página "Índice de acrónimos", en las páginas preliminares (tras el Índice
// de Tablas), con el mismo estilo de encabezado que el resto de índices.
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
