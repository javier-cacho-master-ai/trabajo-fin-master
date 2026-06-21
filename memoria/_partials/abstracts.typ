#import "styles.typ": styles

#let get-abstracts(
  abstract_es: none,
  abstract_en: none,
  keywords: (),
) = {
  // Roman page numbering continues from the indices (no counter reset)
  page(
    numbering: "I",
    number-align: bottom + center,
  )[
    #set heading(numbering: none)
    #set par(..styles.body-paragraph-config)

    = Resumen
    #abstract_es

    #if keywords.len() > 0 [
      #strong[Palabras clave:] #keywords.join(", ")
    ]

    #pagebreak(weak: true)

    = Abstract
    #abstract_en

    #if keywords.len() > 0 [
      #strong[Keywords:] #keywords.join(", ")
    ]
  ]
}
