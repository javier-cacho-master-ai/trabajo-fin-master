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
    #show outline.entry: set block(spacing: styles.outline-entry-spacing)
    #set outline.entry(fill: styles.outline-fill)
    #outline(
      title: none,
      target: figure.where(kind: image).or(figure.where(kind: "quarto-float-fig")),
    )
  ]
}
