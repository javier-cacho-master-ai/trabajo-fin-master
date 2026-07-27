-- Extracts the "Organización del trabajo en grupo" front-matter section
-- (intro text + table) from the body into metadata, the same way
-- process_abstracts.lua relocates the Resumen/Abstract sections.
-- _partials/group-work.typ renders the extracted content with the styling
-- required by the grupal template (see resources/plantilla_grupal_mia.pdf,
-- pages VII-VIII).
local TARGET_HEADING = "Organización del trabajo en grupo"

local function is_h2(el)
  return el.t == "Header" and el.level == 2
end

local function is_target_heading(el)
  return is_h2(el) and pandoc.utils.stringify(el) == TARGET_HEADING
end

local function is_table(el)
  return el.t == "Table"
end

local function slice(blocks, first, last)
  local result = pandoc.List({})
  for i = first, last do
    result:insert(blocks[i])
  end
  return result
end

-- Plain Markdown tables have no row groups, so all data rows live in the
-- single tbl.bodies[1].
local function row_to_meta(row)
  return pandoc.MetaMap({
    apartado = pandoc.MetaString(pandoc.utils.stringify(row.cells[1].contents)),
    responsables = pandoc.MetaString(pandoc.utils.stringify(row.cells[2].contents)),
  })
end

local function table_to_rows(tbl)
  return tbl.bodies[1].body:map(row_to_meta)
end

function Pandoc(doc)
  local blocks = doc.blocks
  local _, section_start = blocks:find_if(is_target_heading)

  if not section_start then
    return doc
  end

  local rest = slice(blocks, section_start + 1, #blocks)
  local _, next_heading_offset = rest:find_if(is_h2)
  local section_end = next_heading_offset and (section_start + next_heading_offset) or (#blocks + 1)

  local before = slice(blocks, 1, section_start - 1)
  local after = slice(blocks, section_end, #blocks)
  doc.blocks = before .. after

  local content = slice(blocks, section_start + 1, section_end - 1)
  local table_block = content:find_if(is_table)
  local intro_blocks = content:filter(function(el) return not is_table(el) end)

  if table_block then
    doc.meta['group_work'] = pandoc.MetaList(table_to_rows(table_block))
  end
  if #intro_blocks > 0 then
    doc.meta['group_work_intro'] = pandoc.MetaBlocks(pandoc.Blocks(intro_blocks))
  end

  return doc
end
