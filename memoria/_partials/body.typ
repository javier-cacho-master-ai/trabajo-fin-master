#import "header.typ": get-header
#import "styles.typ": styles

#let get-body(
  header-text: none,
  content: none,
) = {
  page(
    header: get-header(header-text: header-text),
    numbering: "1",
    number-align: bottom + right,
  )[
    #counter(page).update(1)

    #set par(..styles.body-paragraph-config)
    #set table(align: center + horizon)

    #show figure: set align(center)
    // Align the caption to the right
    #show figure.caption: set align(left)

    #show math.equation: set text(weight: 400)

    // Figures, tables and equations numbered per chapter ("Figura 2.1"),
    // as \numberwithin{...}{chapter} in the LaTeX template
    #set figure(numbering: (..n) => numbering("1.1", counter(heading).get().first(), ..n))
    #set math.equation(numbering: (..n) => numbering("(1.1)", counter(heading).get().first(), ..n))

    #show heading.where(level: 1): it => {
      counter(figure.where(kind: image)).update(0)
      counter(figure.where(kind: table)).update(0)
      counter(figure.where(kind: "quarto-float-fig")).update(0)
      counter(figure.where(kind: "quarto-float-tbl")).update(0)
      counter(math.equation).update(0)
      it
    }

    #content
  ]
}
