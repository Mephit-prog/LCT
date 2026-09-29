import type { RouteLocationRaw } from 'vue-router'
import type { ScanResult } from '@/services/types/api-types'

export type ScreenTarget =
  | { screen: 'wine', slug: string, scanId: string, shouldOfferCandidates: boolean }
  | { screen: 'similar', scanId: string, reason: 'not_found' | 'contract_violation' }

export const resolveScreen = (result: ScanResult): ScreenTarget => {
  const { match, wine, scanId } = result
  if (match.status === 'not_found') {
    return { screen: 'similar', scanId, reason: 'not_found' }
  }
  const slug = wine?.slug ?? match.top1?.slug ?? null
  if (slug === null) {
    return { screen: 'similar', scanId, reason: 'contract_violation' }
  }
  return {
    screen: 'wine',
    slug,
    scanId,
    shouldOfferCandidates: match.status === 'uncertain' && result.candidates.length > 0
  }
}

export const toRouteLocation = (target: ScreenTarget): RouteLocationRaw => {
  if (target.screen === 'similar') {
    return { name: 'similar', params: { scanId: target.scanId } }
  }
  return { name: 'wine', params: { slug: target.slug }, query: { scan: target.scanId } }
}
