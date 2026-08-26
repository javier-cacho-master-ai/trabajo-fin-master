#show: unir_master-ia_tfe_template.with(
  lang: "$lang$",
  title: "$title$",
  subtitle: [$subtitle$],
  author: [$author$],
  director: [$director$],
  work_type: [$work_type$],
  degree: [$degree$],
  date: [$date$],

  abstract_es: [$abstract_es$],
  abstract_en: [$abstract_en$],

  keywords: (
    $for(keywords)$
      "$keywords$",
    $endfor$
  ),
  keywords_en: (
    $for(keywords_en)$
      "$keywords_en$",
    $endfor$
  ),

  group_work_intro: [$group_work_intro$],
  group_work_rows: (
    $for(group_work)$
      [$group_work.apartado$], [$group_work.responsables$],
    $endfor$
  ),
)
