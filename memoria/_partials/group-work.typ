#import "table-of-contents.typ": index-heading
#import "header.typ": get-header

// "Organización del trabajo en grupo" front-matter chapter, required for
// group TFEs (see memoria/resources/plantilla_grupal_mia.pdf, pages VII-VIII).
// Not listed in the Índice de Contenidos, same as the other front-matter
// chapters. `rows` is the flat cell sequence extracted from the Markdown
// table in main.qmd by _filters/process_group_work.lua.
#let get-group-work(header-text: none, rows: ()) = {
  page(
    header: get-header(header-text: header-text),
    numbering: "I",
    number-align: bottom + right,
  )[
    #set heading(numbering: none)
    #index-heading[Organización del trabajo en grupo]

    En este apartado se detallan las distintas partes en las que se ha dividido
    el trabajo entre los componentes del grupo y los mecanismos de coordinación
    empleados.

    #figure(
      table(
        columns: (1fr, auto),
        stroke: 0.5pt + gray,
        inset: 8pt,
        table.header(
          table.cell(colspan: 2, fill: rgb(0, 150, 204), text(fill: white, weight: "bold")[
            Organización del trabajo en grupo - Desarrollo de la memoria
          ]),
          [*Apartado de la memoria*], [*Responsables*],
        ),
        ..rows,
      ),
      caption: [Organización del trabajo en grupo.],
    )
  ]
}
