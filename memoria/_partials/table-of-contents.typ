#import "styles.typ": styles
#import "header.typ": get-header

// Index headings look like chapters but are not numbered nor listed in the TOC
#let index-heading(title) = heading(level: 1, numbering: none, outlined: false)[#title]

#let table-of-contents(header-text: none) = {
  page(
    header: get-header(header-text: header-text),
    numbering: "I",
    number-align: bottom + right,
  )[
    #index-heading[Índice de contenidos]
    #show outline.entry: set block(spacing: styles.outline-entry-spacing)
    #set outline.entry(fill: styles.outline-fill)
    #outline(title: none, depth: 2)
  ]
}
