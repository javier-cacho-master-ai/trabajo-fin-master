#import "styles.typ": styles
#import "header.typ": get-header
#import "table-of-contents.typ": index-heading

#let get-abstracts(
  header-text: none,
  abstract_es: none,
  abstract_en: none,
  keywords: (),
) = {
  // First roman-numbered page after the cover; the cover itself counts as
  // page I even though its number isn't shown (matches the reference, where
  // Resumen is page II).
  page(
    header: get-header(header-text: header-text),
    numbering: "I",
    number-align: bottom + right,
  )[
    #set par(..styles.body-paragraph-config)

    #index-heading[Resumen]
    #abstract_es

    #if keywords.len() > 0 [
      #strong[Palabras clave:] #keywords.join(", ")
    ]

    #pagebreak(weak: true)

    #index-heading[Abstract]
    #abstract_en

    #if keywords.len() > 0 [
      #strong[Keywords:] #keywords.join(", ")
    ]
  ]
}
