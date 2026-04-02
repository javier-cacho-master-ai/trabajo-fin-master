#let styles = {
  let primary_color = rgb("#033142")

  (
    primary_color: primary_color,
    page_config: (
      paper: "a4",
      margin: (x: 2cm, y: 4cm),
    ),
    text_config: (size: 12pt, font: "Calibri"),
    heading_config: (numbering: "1.1."),
    heading_rules_setup: it => {
      set text(fill: primary_color)
      block(above: 2em, below: 1.5em)[
        #it
      ]
    },
    body_paragraph_config: (
      justify: true,
      leading: 1em,
      spacing: 1.2em,
      first-line-indent: 0em,
    ),
  )
}
