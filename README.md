# UNIR - Master IA - Trabajo de Fin de Estudios

Este repositorio contiene la memoria y el código para el Trabajo de Fin de Estudios (TFE) del Master en Inteligencia Artificial (IA) del curso 2025-2026.   

El trabajo se centra en la mejora de la señal astrómica mediante IA.

## Memoria

El fichero fundamental donde se incluye todo el contenido de la memoria 

La estructura del directorio conteniendo la memoria se muestra a continuación con una pequeña explicación de cada ítem.

```sh
│   apa.csl # Fichero de especificación del tipo de cita de las referencias.
│   main.pdf # Fichero de salida generada a partir de main.qmd
│   main.qmd # Fichero con el contenido principal. Este es el que se debe editar.
│   notes.md # Notas rápidas de conceptos útiles
│   references.bib # Referencias bibliográficas.
│   typst-show.typ # Fichero en Typst que pasa los parámetros a la plantilla.
│   typst-template.typ # Plantilla para dar forma al contenido.
│   _quarto.yml # Fichero de configuración de quarto para typst. 
│
├───.quarto # Directorio interno autogenerado por Quarto
├───assets # Directorio para almacenar las figuras usadas en quarto.
├───notes_files # Directory autogenerado por quarto.
├───typst-math # Extensión de typst para el uso de expresiones matemáticas.
│       typst-math.lua
│       _extension.yml
│
└───_partials # Plantillas de formato partial para cada sección.
    │   body.typ
    │   cover.typ
    │   header.typ
    │   styles.typ
    │   table-of-contents.typ
    │
    └───assets # Imágenes y recursos usados por las plantillas parciales.
            logo_unir_1666x348.png
```