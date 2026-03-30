#import "_partials/styles.typ": styles
#import "_partials/cover.typ": cover
#import "_partials/abstracts.typ": get_abstracts
#import "_partials/table-of-contents.typ": table_of_contents
#import "_partials/body.typ": get_body

#let unir_master-ia_tfe_template(
  lang: none,
  title: none,
  subtitle: none,
  author: none,
  abstract_es: none,
  abstract_en: none,
  keywords: none,
  faculty-color: rgb(32, 130, 192),
  date: datetime.today(),
  body,
) = {
  set page(..styles.page_config)
  set text(lang: lang, ..styles.text_config)

  set heading(..styles.heading_config)
  show heading: it => (styles.heading_rules_setup)(it)

  cover(
    title: title,
    author: author,
  )

  get_abstracts(
    abstract_es: abstract_es,
    abstract_en: abstract_en
  )

  table_of_contents()

  get_body(
    header_text: [#author \ #title],
    content: body,
  )
}
