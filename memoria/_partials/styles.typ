#let styles = {
  // "azulunir" from the UNIR LaTeX template: \definecolor{azulunir}{rgb}{0,0.59,0.80}
  let primary-color = rgb(0, 150, 204)
  (
    primary-color: primary-color,
    page-config: (
      paper: "a4",
      margin: (x: 3cm, top: 3.5cm, bottom: 3cm),
    ),
    // 11pt as in \documentclass[11pt,a4paper,spanish]{book}
    text-config: (size: 11pt, font: "Calibri"),
    heading-config: (numbering: "1.1."),
    heading-rules-setup: it => {
      if it.level == 1 {
        // Chapters: \Huge bold azulunir, "1. Title", one per page
        if it.numbering != none and counter(heading).get().first() > 1 {
          pagebreak(weak: true)
        }
        set text(size: 25pt, weight: "bold", fill: primary-color)
        block(above: 0em, below: 2.5em)[#it]
      } else {
        // Sections/subsections: bold black, LaTeX \Large / \large sizes
        let size = if it.level == 2 { 14pt } else { 12pt }
        set text(size: size, weight: "bold")
        block(above: 2em, below: 1em)[#it]
      }
    },
    body-paragraph-config: (
      justify: true,
      // \renewcommand{\baselinestretch}{1.5} at 11pt
      leading: 0.8em,
      // book class: no extra space between paragraphs, first line indented
      spacing: 0.8em,
      first-line-indent: 1.5em,
    ),
  )
}
