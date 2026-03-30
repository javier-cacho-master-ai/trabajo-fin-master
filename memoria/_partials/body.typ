#import "header.typ": get_header
#import "styles.typ": styles

#let get_body(
  header_text: none,
  content: none
) = {
  page(
    header: get_header(header_text: header_text),
    numbering: "1",
    number-align: bottom + right,
  )[
    #counter(page).update(1)

    #set par(..styles.body_paragraph_config)
    #set table(align: center + horizon)

    #show figure: set align(center)
    #show math.equation: set text(weight: 400)

    #content
  ]
}
