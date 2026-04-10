#import "header.typ": get-header
#import "styles.typ": styles

#let get-abstracts(
  header-text: none,
  abstract_es: none,
  abstract_en: none,
) = {
  page(
    header: get-header(header-text: header-text),
    numbering: "I",
    number-align: bottom + right,
  )[
    #counter(page).update(1)

    #set heading(numbering: none, outlined: false)
    #set par(..styles.body-paragraph-config)
    #set table(align: center + horizon)

    = Resumen
    #abstract_es

    #pagebreak(weak: true)

    = Abstract
    #abstract_en
  ]
}
