#import "styles.typ": styles

#let get-header(
  header-text: "header text",
) = {
  set text(
    ..styles.text-config,
    size:10pt,
    weight: "light"
  )
  align(right)[
    #header-text
  ]
}
