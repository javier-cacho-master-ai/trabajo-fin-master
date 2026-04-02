#import "header.typ": get_header
#import "styles.typ": styles

#let get_abstracts(
  header_text: none,
  abstract_es: none,
  abstract_en: none,
) = {
  page(
    header: get_header(header_text: header_text),
    numbering: "I",
    number-align: bottom + right,
  )[
    #counter(page).update(1)

    #set heading(numbering: none)
    #set par(..styles.body_paragraph_config)
    #set table(align: center + horizon)

    = Resumen
    #abstract_es

    #pagebreak()

    = Abstract
    #abstract_en
  ]
}
