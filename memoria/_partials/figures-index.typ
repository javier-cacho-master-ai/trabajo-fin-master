#import "styles.typ": styles
#import "table-of-contents.typ": index-heading
#import "header.typ": get-header

#let get-figures-index(header-text: none) = {
  page(
    header: get-header(header-text: header-text),
    numbering: "I",
    number-align: bottom + right,
  )[
    #index-heading[Índice de figuras]
    #set outline.entry(fill: styles.outline-fill)

    // El bloque envolvente aporta el espaciado entre entradas
    #show outline.entry: it => block(
      spacing: styles.outline-entry-spacing,
      link(
        it.element.location(),
        it.indented(
          it.prefix(),
          emph(it.body()) + [ ] + box(width: 1fr, it.fill) + [ ] + it.page(),
        ),
      ),
    )
    #outline(
      title: none,
      target: figure.where(kind: image).or(figure.where(kind: "quarto-float-fig")),
    )
  ]
}
