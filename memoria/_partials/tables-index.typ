#import "styles.typ": styles
#import "table-of-contents.typ": index-heading
#import "header.typ": get-header

#let get-tables-index(header-text: none) = {
  page(
    header: get-header(header-text: header-text),
    numbering: "I",
    number-align: bottom + right,
  )[
    #index-heading[Índice de tablas]
    #show outline.entry: set block(spacing: styles.outline-entry-spacing)
    #set outline.entry(fill: styles.outline-fill)
    #outline(
      title: none,
      target: figure.where(kind: table).or(figure.where(kind: "quarto-float-tbl")),
    )
  ]
}
