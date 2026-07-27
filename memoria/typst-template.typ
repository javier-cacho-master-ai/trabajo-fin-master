#import "_partials/styles.typ": styles
#import "_partials/cover.typ": cover
#import "_partials/abstracts.typ": get-abstracts
#import "_partials/table-of-contents.typ": table-of-contents
#import "_partials/figures-index.typ": get-figures-index
#import "_partials/tables-index.typ": get-tables-index
#import "_partials/acronyms.typ": acr, get-acronyms-index
#import "_partials/group-work.typ": get-group-work
#import "_partials/body.typ": get-body

#let unir_master-ia_tfe_template(
  lang: none,
  title: none,
  subtitle: none,
  author: none,
  director: none,
  work_type: none,
  degree: none,
  date: none,
  abstract_es: none,
  abstract_en: none,
  keywords: (),
  group_work_rows: (),
  body,
) = {
  set page(..styles.page-config)
  set text(lang: lang, ..styles.text-config)

  set heading(..styles.heading-config)
  show heading: it => (styles.heading-rules-setup)(it)

  // "Figura 1. Caption" / "Tabla 1. Caption", not Typst's default "Figura 1: …"
  set figure.caption(separator: [. ])

  // Table captions go above the table and left-aligned (figure captions stay
  // at their default bottom/centered). Covers both native Typst tables and
  // Quarto-rendered tables (kind: "quarto-float-tbl").
  let table-figure = figure.where(kind: table).or(figure.where(kind: "quarto-float-tbl"))
  show table-figure: set figure.caption(position: top)
  show table-figure: set align(left)

  // Same header on every page after the cover: student name + work title
  let header-text = [#author \ #title]

  cover(
    title: title,
    degree: degree,
    author: author,
    director: director,
    work_type: work_type,
    date: date,
  )

  // Front matter order as in the UNIR grupal template: Resumen/Abstract,
  // then the indices, then "Organización del trabajo en grupo" — all with
  // roman page numbers.
  get-abstracts(
    header-text: header-text,
    abstract_es: abstract_es,
    abstract_en: abstract_en,
    keywords: keywords,
  )

  table-of-contents(header-text: header-text)
  get-figures-index(header-text: header-text)
  get-tables-index(header-text: header-text)

  // Índice de acrónimos, right after the Índice de Tablas (last index above).
  get-acronyms-index(header-text: header-text)

  get-group-work(header-text: header-text, rows: group_work_rows)

  get-body(
    header-text: header-text,
    content: body,
  )
}
