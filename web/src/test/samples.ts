import type { MatchStatus, ScanResult, Wine, WineSummary } from '@/services/types/api-types'

const placeholder = { url: '/bottles/placeholder.svg', altText: null }

export const sampleSummary = (slug: string, title: string, color: string | null = null): WineSummary => ({
  slug,
  title,
  image: placeholder,
  manufacturer: 'Винодельня',
  region: 'Кубань',
  category: color === 'Белое' ? 'Белое сухое' : 'Красное сухое',
  color,
  publicRating: null
})

export const sampleWine = (overrides: Partial<Wine> = {}): Wine => ({
  slug: 'sample-wine',
  title: 'Саперави',
  image: placeholder,
  manufacturer: { name: 'Фанагория', slug: 'fanagoria' },
  region: { name: 'Кубань. Таманский полуостров', image: null },
  grapes: [{ name: 'Саперави', image: null }],
  category: { name: 'Красное сухое', backgroundGradient: null },
  color: 'Красное',
  alcohol: 13.5,
  temperature: '16–18',
  description: null,
  dishes: [{ name: 'Красное мясо', image: null }],
  publicRating: null,
  roskachestvoRating: null,
  similar: [],
  ...overrides
})

const matchOf = (status: MatchStatus, slug: string | null): ScanResult['match'] => ({
  status,
  top1: slug === null ? null : { slug, confidence: 0.9 },
  top5: slug === null ? [] : [{ slug, confidence: 0.9 }],
  margin: 0.4,
  f1Top1: 0.8,
  f1Top5: 0.9,
  latencyMs: 900
})

export const sampleScan = (status: MatchStatus, scanId: string): ScanResult => {
  const wine = status === 'not_found' ? null : sampleWine()
  return {
    scanId,
    match: matchOf(status, wine?.slug ?? 'similar-wine'),
    wine,
    candidates: status === 'uncertain' ? [sampleSummary('other-wine', 'Шардоне', 'Белое')] : [],
    similar: [{ wine: sampleSummary('similar-wine', 'Похожее', 'Красное'), reasons: ['similar_label', 'same_grape'] }],
    alternatives: status === 'not_found'
      ? [{ wine: sampleSummary('alt-wine', 'Альтернатива', 'Красное'), reasons: ['other_manufacturer'] }]
      : [],
    recognized: {
      manufacturer: 'Винодельня',
      grapes: ['Каберне Совиньон'],
      category: 'Красное сухое',
      year: 2022,
      region: 'Кубань'
    }
  }
}
