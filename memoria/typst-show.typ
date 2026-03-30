#show: unir_master-ia_tfe_template.with(
  lang: "$lang$",
  title: "$title$",
  subtitle: [$subtitle$],
  author: "$author$",

  abstract_es: "$abstract_es$",
  abstract_en: "$abstract_en$",
  // abstract: [$abstract$],
  keywords: (
    $for(keywords)$
      "$keywords$",
    $endfor$
  )
)