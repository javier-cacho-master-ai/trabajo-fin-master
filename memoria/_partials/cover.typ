#import "styles.typ": styles

#let cover(
  title: "Title",
  author: "Author",
  director: "Director",
) = {
  set text(fill: styles.primary-color)

  page(
    numbering: none,
  )[
    #align(top + center)[
      #image("assets/logo_unir_1666x348.png", width: 10.7cm)

      #text(size: 24pt)[
        Universidad Internacional de La Rioja
      ]

      #text(size: 20pt)[
        Escuela Superior de Ingeniería y Tecnología
      ]
    ]

    #align(horizon + center)[
      #text(size: 18pt)[
        Máster Universitario en Inteligencia Artificial
      ]

      #text(size: 26pt)[
        #title
      ]
    ]

    #align(bottom + center)[
      #set text(12pt)
      #grid(
        columns: (12em, 12em),
        gutter: 1.2em,

        align: top + left,
        
        [Trabajo fin de estudio presentado por:], [#author],
        
        [Tipo de trabajo], [Desarrollo de Software],
        
        [Director], [#director],
        
        [Fecha], [#datetime.today().display()],
      )
    ]
  ]
}
