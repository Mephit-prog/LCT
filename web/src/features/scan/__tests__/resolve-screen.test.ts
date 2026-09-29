import { describe, expect, it } from 'vitest'
import { sampleScan } from '@/test/samples'
import { resolveScreen, toRouteLocation } from '../resolve-screen'

describe('resolveScreen', () => {
  it('exact → карточка без шторки', () => {
    const result = sampleScan('exact', 'scan-1')
    const target = resolveScreen(result)
    expect(target.screen).toBe('wine')
    if (target.screen === 'wine') {
      expect(target.slug).toBe(result.wine?.slug)
      expect(target.shouldOfferCandidates).toBe(false)
    }
  })

  it('uncertain → карточка top-1 с предложением кандидатов', () => {
    const result = sampleScan('uncertain', 'scan-2')
    const target = resolveScreen(result)
    expect(target).toMatchObject({ screen: 'wine', slug: result.wine?.slug, shouldOfferCandidates: true })
  })

  it('not_found → экран похожих', () => {
    const result = sampleScan('not_found', 'scan-3')
    expect(resolveScreen(result)).toEqual({ screen: 'similar', scanId: 'scan-3', reason: 'not_found' })
  })

  it('exact без wine и top1 → экран похожих как деградация контракта', () => {
    const result = sampleScan('exact', 'scan-4')
    const broken = { ...result, wine: null, match: { ...result.match, top1: null } }
    expect(resolveScreen(broken)).toEqual({ screen: 'similar', scanId: 'scan-4', reason: 'contract_violation' })
  })

  it('формирует маршруты с сохранением scanId', () => {
    expect(toRouteLocation({ screen: 'wine', slug: 'a', scanId: 's', shouldOfferCandidates: false })).toEqual({
      name: 'wine',
      params: { slug: 'a' },
      query: { scan: 's' }
    })
    expect(toRouteLocation({ screen: 'similar', scanId: 's', reason: 'not_found' })).toEqual({
      name: 'similar',
      params: { scanId: 's' }
    })
  })
})
