function Pandoc(doc)
  -- 1. Define storage bins (our state)
  local bins = { main = {}, es = {}, en = {} }
  local current_bin = bins.main
  
  -- 2. Define routing map: links heading text to its respective bin
  local routes = { ["Resumen"] = bins.es, ["Abstract"] = bins.en }

  -- 3. Single-pass declarative loop
  for _, el in ipairs(doc.blocks) do
    local is_h2 = (el.t == "Header" and el.level == 2)
    local title = is_h2 and pandoc.utils.stringify(el) or ""
    local target_route = routes[title]

    -- State Transition: If H2, route to matched bin or reset to main. Otherwise, keep current state.
    current_bin = is_h2 and (target_route or bins.main) or current_bin
    
    -- Action: Insert the element, unless it's one of the routing headers themselves
    if not (is_h2 and target_route) then
      table.insert(current_bin, el)
    end
  end

  -- 4. Reconstruct the document and assign metadata
  doc.blocks = bins.main
  
  if #bins.es > 0 then doc.meta['abstract_es'] = pandoc.MetaBlocks(bins.es) end
  if #bins.en > 0 then doc.meta['abstract_en'] = pandoc.MetaBlocks(bins.en) end

  return doc
end