import type { SommelierAnswers, SommelierMode, SommelierRecommendation, Wine, WineSummary } from '@/services/types/api-types'

export interface SommelierContext {
  mode: SommelierMode
  wine: Wine | null
  candidates: WineSummary[]
}

export interface SommelierReply {
  text: string
  temperature: string | null
  recommendations: SommelierRecommendation[]
}

const dishKeywords: Record<string, string[]> = {
  meat: ['мяс', 'стейк', 'говяд', 'баран', 'дич'],
  poultry: ['птиц', 'кур', 'утк', 'индей'],
  fish: ['рыб', 'лосос', 'форел'],
  seafood: ['морепродукт', 'креветк', 'устриц'],
  cheese: ['сыр'],
  vegetables: ['овощ', 'гриль', 'салат', 'зелен'],
  dessert: ['десерт', 'сладк', 'фрукт', 'шоколад']
}

const dishPhrases: Record<string, string> = {
  meat: 'к мясу',
  poultry: 'к птице',
  fish: 'к рыбе',
  seafood: 'к морепродуктам',
  cheese: 'к сырам',
  vegetables: 'к овощам',
  dessert: 'к десерту'
}

const occasionPhrases: Record<string, string> = {
  dinner: 'для ужина дома',
  gift: 'в подарок',
  celebration: 'для праздника',
  try: 'чтобы просто попробовать'
}

const dishColorFit: Record<string, string[]> = {
  meat: ['Красное'],
  poultry: ['Белое', 'Розовое', 'Красное'],
  fish: ['Белое', 'Розовое'],
  seafood: ['Белое', 'Розовое'],
  cheese: ['Белое', 'Красное'],
  vegetables: ['Розовое', 'Белое'],
  dessert: ['Белое']
}

const colorDefaultDishes: Record<string, string> = {
  'Красное': 'к мясу и выдержанным сырам',
  'Белое': 'к рыбе, птице и молодым сырам',
  'Розовое': 'к овощам на гриле, морепродуктам и лёгким закускам'
}

export type TasteCode = 'light' | 'rich' | 'sweet'
type Sweetness = 'dry' | 'offDry' | 'semiSweet' | 'sweet' | 'brut'
type StyleVerdict = 'yes' | 'no' | 'unknown'

interface StyleSource {
  title: string
  category: string
  color: string | null
  alcohol: number | null
}

const tastePhrases: Record<TasteCode, string> = {
  light: 'лёгкое и свежее',
  rich: 'насыщенное',
  sweet: 'сладкое'
}

const sweetnessNames: Record<Sweetness, string> = {
  dry: 'сухое',
  offDry: 'полусухое',
  semiSweet: 'полусладкое',
  sweet: 'сладкое',
  brut: 'брют'
}

export const isDishCode = (value: string | null): value is string => value !== null && value in dishKeywords

const dishSearchWords: Record<string, string[]> = {
  meat: ['красное'],
  poultry: ['белое', 'красное'],
  fish: ['белое', 'розовое'],
  seafood: ['белое', 'розовое'],
  cheese: ['красное', 'белое'],
  vegetables: ['белое', 'розовое'],
  dessert: ['белое']
}

export const dishSearchQueries = (dish: string): string[] => dishSearchWords[dish] ?? []

export const isTasteCode = (value: string | null): value is TasteCode => {
  return value === 'light' || value === 'rich' || value === 'sweet'
}

const detectSweetness = (category: string, title: string): Sweetness | null => {
  const text = `${category} ${title}`.toLocaleLowerCase('ru')
  if (text.includes('полуслад')) {
    return 'semiSweet'
  }
  if (text.includes('полусух')) {
    return 'offDry'
  }
  if (text.includes('сладк')) {
    return 'sweet'
  }
  if (text.includes('брют')) {
    return 'brut'
  }
  if (text.includes('сух')) {
    return 'dry'
  }
  return null
}

const tasteFit = (source: StyleSource, taste: TasteCode): StyleVerdict => {
  if (taste === 'sweet') {
    const sweetness = detectSweetness(source.category, source.title)
    if (sweetness === 'sweet' || sweetness === 'semiSweet') {
      return 'yes'
    }
    if (sweetness === null) {
      return 'unknown'
    }
    return 'no'
  }
  if (taste === 'light') {
    if (source.alcohol !== null && source.alcohol >= 14) {
      return 'no'
    }
    if (source.color === 'Красное') {
      return 'no'
    }
    if (source.color === 'Белое' || source.color === 'Розовое') {
      return 'yes'
    }
    if (source.alcohol !== null && source.alcohol <= 12) {
      return 'yes'
    }
    return 'unknown'
  }
  if (source.alcohol !== null && source.alcohol >= 13.5) {
    return 'yes'
  }
  if (source.color === 'Красное') {
    return 'yes'
  }
  if (source.color === 'Белое' || source.color === 'Розовое') {
    return 'no'
  }
  return 'unknown'
}

const sourceOfWine = (wine: Wine): StyleSource => ({
  title: wine.title,
  category: wine.category.name,
  color: wine.color,
  alcohol: wine.alcohol
})

const sourceOfSummary = (wine: WineSummary): StyleSource => ({
  title: wine.title,
  category: wine.category,
  color: wine.color,
  alcohol: null
})

const findMatchingDish = (wine: Wine, dish: string): string | null => {
  const keywords = dishKeywords[dish] ?? []
  const match = wine.dishes.find((item) => {
    const name = item.name.toLowerCase()
    return keywords.some((keyword) => name.includes(keyword))
  })
  return match?.name ?? null
}

const sameColor = (color: string, expected: string): boolean => {
  return color.trim().toLocaleLowerCase('ru') === expected.toLocaleLowerCase('ru')
}

const fitsByColor = (color: string | null, dish: string | null): boolean => {
  if (dish === null) {
    return true
  }
  const allowed = dishColorFit[dish]
  if (allowed === undefined) {
    return true
  }
  if (color === null || color.trim() === '') {
    return false
  }
  return allowed.some((expected) => sameColor(color, expected))
}

const listDishes = (wine: Wine): string => {
  return wine.dishes.slice(0, 3).map((item) => item.name.toLowerCase()).join(', ')
}

const temperatureSentence = (wine: Wine): string => {
  if (wine.temperature === null || wine.temperature.trim() === '') {
    return ''
  }
  return ` Подавайте при ${wine.temperature} °C.`
}

const pickRecommendations = (pool: WineSummary[], exclude: string | null, dish: string | null, reason: string, limit = 3): SommelierRecommendation[] => {
  const seen = new Set<string>()
  const result: SommelierRecommendation[] = []
  for (const wine of pool) {
    if (wine.slug === exclude || seen.has(wine.slug) || !fitsByColor(wine.color, dish)) {
      continue
    }
    seen.add(wine.slug)
    result.push({ wine, reason })
    if (result.length === limit) {
      break
    }
  }
  if (result.length > 0 || dish !== null) {
    return result
  }
  return pool
    .filter((wine) => wine.slug !== exclude)
    .slice(0, limit)
    .map((wine) => ({ wine, reason: 'похожее по стилю' }))
}

const buildPairingWithDish = (wine: Wine, dish: string): SommelierReply => {
  const dishPhrase = dishPhrases[dish] ?? 'к вашему блюду'
  const matched = findMatchingDish(wine, dish)
  if (matched !== null) {
    return {
      text: `Подойдёт: ${wine.title} создано в том числе под «${matched.toLowerCase()}». ${wine.category.name} из сорта ${wine.grapes.map((grape) => grape.name).join(', ') || 'этого купажа'} поддержит вкус блюда, не перебивая его.${temperatureSentence(wine)}`,
      temperature: wine.temperature,
      recommendations: pickRecommendations(wine.similar, wine.slug, dish, `тоже подходит ${dishPhrase}`)
    }
  }
  if (fitsByColor(wine.color, dish)) {
    const dishes = wine.dishes.length > 0 ? ` В каталоге его рекомендуют к: ${listDishes(wine)}.` : ''
    return {
      text: `Подойдёт: ${wine.title} по стилю уместно ${dishPhrase}, хотя это не самое каноничное сочетание.${dishes}${temperatureSentence(wine)}`,
      temperature: wine.temperature,
      recommendations: pickRecommendations(wine.similar, wine.slug, dish, `точнее ${dishPhrase}`)
    }
  }
  const usual = wine.dishes.length > 0 ? listDishes(wine) : (colorDefaultDishes[wine.color ?? ''] ?? 'закускам в его стиле')
  return {
    text: `Лучше взять другое: ${wine.title} — ${wine.category.name.toLowerCase()}, его подают ${usual}. ${dishPhrase.charAt(0).toUpperCase()}${dishPhrase.slice(1)} традиционно берут вина другого цвета — варианты ниже.`,
    temperature: null,
    recommendations: pickRecommendations(wine.similar, wine.slug, dish, `подходит ${dishPhrase}`)
  }
}

export const suggestDishMatches = (wine: Wine, dish: string, pool: WineSummary[]): WineSummary[] => {
  const seen = new Set<string>()
  const result: WineSummary[] = []
  for (const item of pool) {
    if (item.slug === wine.slug || seen.has(item.slug) || !fitsByColor(item.color, dish)) {
      continue
    }
    seen.add(item.slug)
    result.push(item)
    if (result.length === 3) {
      break
    }
  }
  return result
}

export const offerDishAlternative = (reply: SommelierReply, suggestions: WineSummary[], dish: string): SommelierReply => {
  const title = suggestions[0]?.title
  const dishPhrase = dishPhrases[dish] ?? 'к вашему блюду'
  if (title === undefined) {
    return {
      ...reply,
      text: reply.text.replace(' — варианты ниже', '')
    }
  }
  if (reply.recommendations.length > 0 || reply.text.includes('Вместо него возьмите')) {
    return reply
  }
  return {
    ...reply,
    temperature: null,
    recommendations: suggestions.slice(0, 3).map((item) => ({
      wine: item,
      reason: dishPhrase
    })),
    text: `${reply.text} Вместо него возьмите ${title}.`
  }
}

const pickByTaste = (pool: WineSummary[], exclude: string | null, taste: TasteCode, reason: string): SommelierRecommendation[] => {
  return pool
    .filter((item) => item.slug !== exclude && tasteFit(sourceOfSummary(item), taste) === 'yes')
    .slice(0, 3)
    .map((item) => ({ wine: item, reason }))
}

const mismatchReason = (wine: Wine, taste: TasteCode): string => {
  if (taste === 'sweet') {
    const sweetness = detectSweetness(wine.category.name, wine.title)
    if (sweetness === null) {
      return 'в карточке не указана сладость, а вам ближе сладкое'
    }
    return `это ${sweetnessNames[sweetness]} вино, а вам ближе сладкое`
  }
  if (tasteFit(sourceOfWine(wine), taste) === 'unknown') {
    return `по карточке не видно, что оно ${tastePhrases[taste]}`
  }
  if (taste === 'light') {
    return 'оно насыщеннее, чем лёгкое и свежее, которое вам ближе'
  }
  return 'оно легче, чем насыщенное, которое вам ближе'
}

const tastePool = (wine: Wine, extra: WineSummary[]): WineSummary[] => {
  const seen = new Set<string>()
  return [...wine.similar, ...extra].filter((item) => {
    if (item.slug === wine.slug || seen.has(item.slug)) {
      return false
    }
    seen.add(item.slug)
    return true
  })
}

const grapeScore = (item: WineSummary, grape: string, grapes: ReadonlyMap<string, string>): number => {
  if (grape === '') {
    return 0
  }
  const listed = (grapes.get(item.slug) ?? '').split(/[,;]/).map((part) => part.trim().toLocaleLowerCase('ru')).filter((part) => part !== '')
  const haystack = `${item.title} ${item.category}`.toLocaleLowerCase('ru')
  if (listed.includes(grape) || haystack.includes(grape)) {
    return listed.length === 1 ? 2 : 1
  }
  return 0
}

export const suggestTasteMatches = (wine: Wine, taste: TasteCode, pool: WineSummary[], grapes: ReadonlyMap<string, string> = new Map()): WineSummary[] => {
  const grape = wine.grapes[0]?.name.trim().toLocaleLowerCase('ru') ?? ''
  return pool
    .filter((item) => item.slug !== wine.slug && tasteFit(sourceOfSummary(item), taste) === 'yes')
    .sort((left, right) => grapeScore(right, grape, grapes) - grapeScore(left, grape, grapes))
    .filter((item, index, items) => items.findIndex((other) => other.slug === item.slug) === index)
    .slice(0, 3)
}

const offerSentence = (title: string | undefined): string => {
  if (title === undefined) {
    return ''
  }
  return ` Вместо него возьмите ${title}.`
}

export const offerTasteAlternative = (reply: SommelierReply, suggestions: WineSummary[], taste: TasteCode): SommelierReply => {
  const title = suggestions[0]?.title
  if (title === undefined || reply.recommendations.length > 0 || reply.text.includes('Вместо него возьмите')) {
    return reply
  }
  const text = reply.text.replace(/\s*Подавайте при[^.]+\.\s*$/u, '').trim()
  return {
    ...reply,
    temperature: null,
    recommendations: suggestions.slice(0, 3).map((item) => ({
      wine: item,
      reason: tastePhrases[taste]
    })),
    text: `${text} Вместо него возьмите ${title}.`
  }
}

const buildPairingByTaste = (wine: Wine, taste: TasteCode, extra: WineSummary[]): SommelierReply => {
  const verdict = tasteFit(sourceOfWine(wine), taste)
  const recommendations = suggestTasteMatches(wine, taste, tastePool(wine, extra)).map((item) => ({
    wine: item,
    reason: tastePhrases[taste]
  }))
  if (verdict !== 'yes') {
    return {
      text: `Лучше взять другое: ${wine.title} — ${mismatchReason(wine, taste)}.${offerSentence(recommendations[0]?.wine.title)}`.trim(),
      temperature: null,
      recommendations
    }
  }
  const sweetness = detectSweetness(wine.category.name, wine.title)
  const note = taste === 'sweet' && sweetness === 'semiSweet' ? 'полусладкое, близко к сладкому' : tastePhrases[taste]
  return {
    text: `Подойдёт: ${wine.title} — ${note}, чтобы просто попробовать.${temperatureSentence(wine)}`.trim(),
    temperature: wine.temperature,
    recommendations
  }
}

const buildPairingByOccasion = (wine: Wine, answers: SommelierAnswers): SommelierReply => {
  const occasion = occasionPhrases[answers.occasion] ?? 'на ваш случай'
  const ratingNote = wine.publicRating !== null && wine.publicRating >= 4.5
    ? ` Оценка пользователей ${wine.publicRating.toFixed(2)} — надёжный ориентир.`
    : wine.roskachestvoRating !== null
      ? ` Отмечено Роскачеством: ${wine.roskachestvoRating.score} баллов.`
      : ''
  const pairing = wine.dishes.length > 0 ? ` Лучше всего раскроется с: ${listDishes(wine)}.` : ''
  return {
    text: `Подойдёт: ${wine.title} — уверенный выбор ${occasion}.${ratingNote}${pairing}${temperatureSentence(wine)}`.trim(),
    temperature: wine.temperature,
    recommendations: pickRecommendations(wine.similar, wine.slug, null, 'похожее по стилю')
  }
}

const buildDiscovery = (context: SommelierContext, answers: SommelierAnswers): SommelierReply => {
  if (isTasteCode(answers.dish)) {
    const wanted = tastePhrases[answers.dish]
    const occasion = occasionPhrases[answers.occasion] ?? 'на ваш случай'
    const recommendations = pickByTaste(context.candidates, null, answers.dish, wanted)
    const intro = recommendations.length > 0
      ? 'Ниже — вина из подборки ближе к этому вкусу.'
      : 'В подборке нет вина такого вкуса.'
    return {
      text: `Подбираем ${wanted} вино ${occasion}. ${intro}`,
      temperature: null,
      recommendations
    }
  }
  const occasion = occasionPhrases[answers.occasion] ?? 'на ваш случай'
  const dish = isDishCode(answers.dish) ? ` ${dishPhrases[answers.dish ?? ''] ?? ''}` : ''
  const recommendations = pickRecommendations(context.candidates, null, isDishCode(answers.dish) ? answers.dish : null, isDishCode(answers.dish) ? `подходит${dish}` : 'близко к вашему фото')
  const intro = recommendations.length > 0
    ? 'Ниже — вина из подборки, которые ближе всего к вашему запросу.'
    : 'Сфотографируйте другую бутылку или откройте каталог — подберём точнее.'
  return {
    text: `Подбираем вино ${occasion}${dish}. ${intro}`,
    temperature: null,
    recommendations
  }
}

export const buildFallbackReply = (context: SommelierContext, answers: SommelierAnswers): SommelierReply => {
  if (context.mode === 'discovery' || context.wine === null) {
    return buildDiscovery(context, answers)
  }
  if (isDishCode(answers.dish) && answers.dish !== null) {
    return buildPairingWithDish(context.wine, answers.dish)
  }
  if (isTasteCode(answers.dish)) {
    return buildPairingByTaste(context.wine, answers.dish, context.candidates)
  }
  return buildPairingByOccasion(context.wine, answers)
}
