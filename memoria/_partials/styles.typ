#let styles = {
  let primary-color = rgb("#033142")

  (
    primary-color: primary-color,
    page-config: (
      paper: "a4",
      margin: (x: 2cm, y: 4cm),
    ),
    text-config: (size: 12pt, font: "Calibri"),
    heading-config: (numbering: "1.1."),
    heading-rules-setup: it => {
      // Base styling rule (no page break)
      let heading-base-rule(above: 0em) = it => {
        set text(fill: primary-color)
        block(
          above: above,
          below: 1.5em,
        )[#it]
      }
      let above = if it.level == 1 { 0em } else { 2em }

      // Page break only for level 1 and from the 3rd occurrence onward
      if it.level == 1 and counter(heading).get().first() > 2 {
        pagebreak(weak: true)
      }

      heading-base-rule(above: above)(it)
    },
    body-paragraph-config: (
      justify: true,
      leading: 1em,
      spacing: 1.2em,
      first-line-indent: 0em,
    ),
  )
}
