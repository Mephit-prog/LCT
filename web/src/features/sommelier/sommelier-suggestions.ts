import { apis } from '@/services/apis'
import { toWineSummary } from '@/services/apis/map-wine'
import type { Wine, WineSummary } from '@/services/types/api-types'
import { dishSearchQueries, suggestDishMatches, suggestTasteMatches, type TasteCode } from './sommelier-fallback'

const searchQueries = (wine: Wine, taste: TasteCode): string[] => {
  const grape = wine.grapes[0]?.name.trim() ?? ''
  const color = wine.color?.trim().toLocaleLowerCase('ru') ?? ''
  const base = [grape, color].filter((part) => part !== '').join(' ')
  if (taste === 'sweet') {
    return [`${base} полусладкое`.trim(), `${base} сладкое`.trim()]
  }
  if (taste === 'light') {
    return [grape === '' ? 'белое' : `${grape} белое`]
  }
  return [grape === '' ? 'красное' : `${grape} красное`]
}

export const findTasteAlternatives = async (wine: Wine, taste: TasteCode, signal: AbortSignal): Promise<WineSummary[]> => {
  const cards = (await Promise.all(searchQueries(wine, taste).map((query) => apis.searchWines(query, signal)))).flat()
  const grapes = new Map(cards.map((card) => [card.slug, card.grape ?? '']))
  return suggestTasteMatches(wine, taste, cards.map((card) => toWineSummary(card)), grapes)
}

export const findDishAlternatives = async (wine: Wine, dish: string, signal: AbortSignal): Promise<WineSummary[]> => {
  const queries = dishSearchQueries(dish)
  if (queries.length === 0) {
    return []
  }
  const cards = (await Promise.all(queries.map((query) => apis.searchWines(query, signal)))).flat()
  return suggestDishMatches(wine, dish, cards.map((card) => toWineSummary(card)))
}
