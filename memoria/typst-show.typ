#show: unir_master-ia_tfe_template.with(
  lang: "$lang$",
  title: "$title$",
  subtitle: [$subtitle$],
  author: [$author$],
  director: [$director$],
  degree: [$degree$],
  city: [$city$],
  date: [$date$],

  abstract_es: [$abstract_es$],
  abstract_en: [$abstract_en$],

  keywords: (
    $for(keywords)$
      "$keywords$",
    $endfor$
  )
)
