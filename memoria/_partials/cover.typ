#import "styles.typ": styles

#let cover(
  title: "Title",
  author: "Author",
) = {
  set text(fill: styles.primary_color)

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
        columns: 2,
        gutter: .8em,
        align: left,
        
        [Trabajo fin de estudio presentado por:], [#author],
        
        [Tipo de trabajo], [Desarrollo de Software],
        
        [Director], [],
        
        [Fecha], [#datetime.today().display()],
      )
    ]
  ]
}
