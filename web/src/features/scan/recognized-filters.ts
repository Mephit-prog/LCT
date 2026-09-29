import type { Recognized, Suggestion } from '@/services/types/api-types'

export type RecognizedFilterKind = 'manufacturer' | 'grape' | 'category' | 'year' | 'region'

export interface RecognizedFilter {
  id: string
  kind: RecognizedFilterKind
  label: string
}

const normalize = (value: string): string => value.trim().toLowerCase()

export const buildRecognizedFilters = (recognized: Recognized | null): RecognizedFilter[] => {
  if (recognized === null) {
    return []
  }
  const filters: RecognizedFilter[] = []
  if (recognized.manufacturer !== null && recognized.manufacturer.trim() !== '') {
    filters.push({ id: 'manufacturer', kind: 'manufacturer', label: recognized.manufacturer })
  }
  recognized.grapes.forEach((grape, index) => {
    if (grape.trim() !== '') {
      filters.push({ id: `grape-${index}`, kind: 'grape', label: grape })
    }
  })
  if (recognized.category !== null && recognized.category.trim() !== '') {
    filters.push({ id: 'category', kind: 'category', label: recognized.category })
  }
  if (recognized.year !== null) {
    filters.push({ id: 'year', kind: 'year', label: String(recognized.year) })
  }
  if (recognized.region !== null && recognized.region.trim() !== '') {
    filters.push({ id: 'region', kind: 'region', label: recognized.region })
  }
  return filters
}

const matchesFilter = (suggestion: Suggestion, filter: RecognizedFilter): boolean => {
  const label = normalize(filter.label)
  if (filter.kind === 'manufacturer') {
    return normalize(suggestion.wine.manufacturer) === label || suggestion.reasons.includes('same_manufacturer')
  }
  if (filter.kind === 'category') {
    return normalize(suggestion.wine.category).includes(label) || suggestion.reasons.includes('same_category')
  }
  if (filter.kind === 'region') {
    return normalize(suggestion.wine.region).includes(label) || suggestion.reasons.includes('same_region')
  }
  if (filter.kind === 'grape') {
    return suggestion.reasons.includes('same_grape')
  }
  return true
}

export const applyRecognizedFilters = (items: Suggestion[], filters: RecognizedFilter[]): Suggestion[] => {
  if (filters.length === 0) {
    return items
  }
  return items.filter((item) => filters.every((filter) => matchesFilter(item, filter)))
}
