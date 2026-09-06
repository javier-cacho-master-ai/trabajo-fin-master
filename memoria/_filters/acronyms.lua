-- Referencias limpias a acrónimos en el texto, sin escribir Typst en el Markdown.
-- Un span `[ESA]{.acr}` se convierte en una llamada al ayudante `acr` definido en
-- `_partials/acronyms.typ`, que desarrolla la primera aparición y muestra solo la
-- sigla en las siguientes. El texto del span es la clave del acrónimo.
-- Añadir `.short` (`[SNR]{.acr .short}`) fuerza la sigla sola y no cuenta como
-- primera aparición: es la forma para encabezados y pies de figura, que se
-- reproducen en los índices preliminares y adelantarían el desarrollo.
function Span(el)
  if el.classes:includes("acr") then
    local key = pandoc.utils.stringify(el.content)
    local short = el.classes:includes("short") and ", short: true" or ""
    return pandoc.RawInline("typst", '#acr("' .. key .. '"' .. short .. ')')
  end
end
