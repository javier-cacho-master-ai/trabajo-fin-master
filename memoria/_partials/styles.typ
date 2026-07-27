#let styles = {
  // "azulunir" from the UNIR LaTeX template: \definecolor{azulunir}{rgb}{0,0.59,0.80}
  let primary-color = rgb(0, 150, 204)
  (
    primary-color: primary-color,
    page-config: (
      paper: "a4",
      margin: (
        top: 3cm,
        bottom: 2.49cm,
        left: 3cm,
        right: 2cm,
      ),
      // header-ascent: 0.6cm,
    ),
    text-config: (size: 12pt, font: "Calibri"),
    heading-config: (numbering: "1.1."),
    heading-rules-setup: it => {
      if (
        it.level == 1 and it.numbering != none and counter(heading).get().first() > 1
      ) {
        pagebreak(weak: true)
      }

      let text-config = (
        size: 12pt,
        weight: "light"
      )
 
      let block-config = (above: 1em, below: 1.5em)

      if it.level == 1 {
        text-config += (
          size: 18pt,
          fill: primary-color
        )
      }

      if it.level == 2 {
        text-config += (
          size: 14pt, 
          fill: primary-color
        )
      }

      if it.level == 3 {
        text-config += (weight: "bold")
      }

      set text(
        ..text-config,
      )
      block(
        ..block-config
      )[#it]
    },
    body-paragraph-config: (
      justify: true,
      leading: 1.5em,
      spacing: 2em,
    ),
  )
}
