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
  abstract_es: none,
  abstract_en: none,
  keywords: none,
  faculty-color: rgb(32, 130, 192),
  date: datetime.today(),
  body,
) = {
  set page(..styles.page-config)
  set text(lang: lang, ..styles.text-config)

  set heading(..styles.heading-config)
  show heading: it => (styles.heading-rules-setup)(it)

  cover(
    title: title,
    author: author,
    director: director,
  )

  get-abstracts(
    abstract_es: abstract_es,
    abstract_en: abstract_en
  )

  table-of-contents()

  get-body(
    header-text: [#author #title],
    content: body,
  )
}
