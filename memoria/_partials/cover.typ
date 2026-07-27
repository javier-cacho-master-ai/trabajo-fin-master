#import "styles.typ": styles

// Reproduces the cover of the UNIR TFE grupal template (see
// memoria/resources/plantilla_grupal_mia.pdf, page 1): centered logo and
// title block, no border, metadata as a table.
#let cover(
  title: none,
  degree: none,
  author: none,
  director: none,
  work_type: none,
  date: none,
) = {
  page(
    numbering: none,
  )[
    #v(2cm)
    #align(center)[
      #image("assets/logo_unir_1666x348.png", width: 10.68cm)

      #v(1cm)
      #text(size: 24pt, weight: "light")[Universidad Internacional de La Rioja]

      // #v(0.1cm)
      #text(size: 20pt, weight: "light")[Escuela Superior de Ingeniería y Tecnología]

      #v(3cm)
      #text(size: 18pt, weight: "light")[#degree]

      // #v(0.3cm)
      #text(size: 26pt, fill: styles.primary-color)[#title]
    ]

    #v(1fr)

    #table(
      columns: (auto, 1fr),
      stroke: 0.5pt + gray,
      inset: 8pt,
      [Trabajo fin de estudio presentado por:], [#author],
      [Tipo de trabajo:], [#work_type],
      [Director:], [#director],
      [Fecha:], [#date],
    )
  ]
}
