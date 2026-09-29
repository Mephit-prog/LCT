import { describe, expect, it } from 'vitest'
import { sampleScan, sampleWine } from '@/test/samples'
import { scanResultSchema, wineSchema } from '../schemas'

describe('схемы API', () => {
  it('полная карточка соответствует модели Wine', () => {
    const wine = sampleWine()
    expect(wineSchema.safeParse(wine).success).toBe(true)
  })

  it('результаты скана для всех статусов валидны', () => {
    for (const scenario of ['exact', 'uncertain', 'not_found'] as const) {
      const result = sampleScan(scenario, `scan-${scenario}`)
      expect(scanResultSchema.safeParse(result).success).toBe(true)
    }
  })

  it('not_found гарантирует непустой similar и wine = null', () => {
    const result = sampleScan('not_found', 'scan')
    expect(result.wine).toBeNull()
    expect(result.similar.length).toBeGreaterThan(0)
  })

  it('отклоняет неизвестный статус и отсутствующие поля', () => {
    const result = sampleScan('exact', 'scan')
    expect(scanResultSchema.safeParse({ ...result, match: { ...result.match, status: 'maybe' } }).success).toBe(false)
    const wine = sampleWine()
    expect(wineSchema.safeParse({ ...wine, title: undefined }).success).toBe(false)
    expect(wineSchema.safeParse({ ...wine, alcohol: 'много' }).success).toBe(false)
  })

  it('допускает null в необязательных полях', () => {
    const wine = sampleWine()
    const sparse = { ...wine, alcohol: null, temperature: null, description: null, publicRating: null, roskachestvoRating: null, color: null }
    expect(wineSchema.safeParse(sparse).success).toBe(true)
  })
})
