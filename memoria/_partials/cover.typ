#import "styles.typ": styles

// Reproduces the \creaportada cover of the UNIR LaTeX template
#let cover(
  title: none,
  degree: none,
  author: none,
  director: none,
  work_type: none,
  city: none,
  date: none,
) = {
  page(
    numbering: none,
  )[
    #align(top + left)[
      #image("assets/logo_unir_1666x348.png", width: 80%)
    ]

    #v(1cm)

    #align(center)[
      #block(
        width: 12cm,
        stroke: (left: 2pt + styles.primary-color),
        inset: (left: 0.8em, top: 0.5em, bottom: 0.5em),
      )[
        #set align(left)
        #set par(justify: false, leading: 0.65em)

        #text(size: 14pt, weight: "bold")[Universidad Internacional de la Rioja (UNIR)]

        #v(0.3cm)
        #text(size: 20pt, weight: "bold")[Escuela Superior de Ingeniería y Tecnología]

        #v(0.3cm)
        #text(size: 14pt, weight: "bold")[#degree]

        #v(1cm)
        #text(size: 25pt, weight: "bold", fill: styles.primary-color)[#title]
      ]
    ]

    #v(1fr)

    #align(left)[
      #strong[Trabajo Fin de Estudios]
      #v(0.1cm)
      #par[#strong[presentado por:] #author]
      #v(0.1cm)
      #par[#strong[Dirigido por:] #director]
      #v(0.1cm)
      #par[#strong[Tipo de Trabajo:] #work_type]
    ]

    #v(2.5cm)

    #par[#text(fill: styles.primary-color)[Ciudad: ]#city]
    #par[#text(fill: styles.primary-color)[Fecha: ]#date]
  ]
}
