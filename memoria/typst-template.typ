#import "_partials/styles.typ": styles
#import "_partials/cover.typ": cover
#import "_partials/abstracts.typ": get-abstracts
#import "_partials/table-of-contents.typ": table-of-contents
#import "_partials/body.typ": get-body

#let unir_master-ia_tfe_template(
  lang: none,
  title: none,
  subtitle: none,
  author: none,
  director: none,
  work_type: none,
  degree: none,
  city: none,
  date: none,
  abstract_es: none,
  abstract_en: none,
  keywords: (),
  body,
) = {
  set page(..styles.page-config)
  set text(lang: lang, ..styles.text-config)

  set heading(..styles.heading-config)
  show heading: it => (styles.heading-rules-setup)(it)

  cover(
    title: title,
    degree: degree,
    author: author,
    director: director,
    work_type: work_type,
    city: city,
    date: date,
  )

  // Front matter order as in the LaTeX template:
  // indices first, then Resumen/Abstract, all with roman page numbers
  table-of-contents()

  get-abstracts(
    abstract_es: abstract_es,
    abstract_en: abstract_en,
    keywords: keywords,
  )

  get-body(
    // \rhead{\theauthor\\\@titulacion}
    header-text: [#author \ #degree],
    content: body,
  )
}
