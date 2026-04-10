#import "styles.typ": styles

#let table-of-contents() = {
  page(
    numbering: none,
  )[
    #outline(depth: 3)
  ]
}
