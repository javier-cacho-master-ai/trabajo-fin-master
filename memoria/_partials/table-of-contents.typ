#import "styles.typ": styles

// Index headings look like chapters but are not numbered nor listed in the TOC
#let index-heading(title) = heading(level: 1, numbering: none, outlined: false)[#title]

#let table-of-contents() = {
  page(
    numbering: "I",
    number-align: bottom + center,
  )[
    #counter(page).update(1)

    // Chapter entries in bold, with extra separation (LaTeX book TOC style)
    #show outline.entry.where(level: 1): it => {
      v(0.9em, weak: true)
      strong(it)
    }

    #index-heading[Índice de Contenidos]
    #outline(title: none, depth: 1)

    #pagebreak(weak: true)
    #index-heading[Índice de Ilustraciones]
    #outline(
      title: none,
      target: figure.where(kind: image).or(figure.where(kind: "quarto-float-fig")),
    )

    #pagebreak(weak: true)
    #index-heading[Índice de Tablas]
    #outline(
      title: none,
      target: figure.where(kind: table).or(figure.where(kind: "quarto-float-tbl")),
    )
  ]
}
