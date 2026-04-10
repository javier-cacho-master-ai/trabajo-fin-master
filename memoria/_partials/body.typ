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
    #show math.equation: set text(weight: 400)

    #content
  ]
}
