-- Clean in-text acronym references without writing Typst in the Markdown.
-- A span `[ESA]{.acr}` becomes a call to the `acr` helper defined in
-- `_partials/acronyms.typ`, which expands the first use (APA 7.ª) and shows just
-- the sigla afterwards. The span's text is the acronym key.
function Span(el)
  if el.classes:includes("acr") then
    local key = pandoc.utils.stringify(el.content)
    return pandoc.RawInline("typst", '#acr("' .. key .. '")')
  end
end
