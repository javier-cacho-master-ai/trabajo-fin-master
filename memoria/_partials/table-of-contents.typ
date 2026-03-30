#import "styles.typ": styles

#let table_of_contents() = {
  page(
    numbering: none,
  )[
    #outline(depth: 3)
  ]
}
