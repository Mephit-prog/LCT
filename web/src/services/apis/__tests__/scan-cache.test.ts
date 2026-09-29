import { beforeEach, describe, expect, it } from 'vitest'
import { readCachedScan, writeCachedScan } from '@/services/apis/scan-cache'
import { storageKeys } from '@/services/constants/storage-keys'
import type { ScanResult } from '@/services/types/api-types'

const result = (scanId: string): ScanResult => ({
  scanId,
  match: {
    status: 'exact',
    top1: { slug: 'fanagoria', confidence: 0.9 },
    top5: [{ slug: 'fanagoria', confidence: 0.9 }],
    margin: null,
    f1Top1: null,
    f1Top5: null,
    latencyMs: 10
  },
  wine: {
    slug: 'fanagoria',
    title: 'Рислинг',
    image: { url: '/bottles/placeholder.svg', altText: 'Рислинг' },
    manufacturer: { name: 'Фанагория', slug: '' },
    region: { name: '', image: null },
    grapes: [],
    category: { name: 'Белое', backgroundGradient: null },
    color: null,
    alcohol: null,
    temperature: null,
    description: null,
    dishes: [],
    publicRating: null,
    roskachestvoRating: null,
    similar: []
  },
  candidates: [],
  similar: [],
  alternatives: [],
  recognized: null
})

describe('кэш скана', () => {
  beforeEach(() => {
    sessionStorage.clear()
  })

  it('читает записанный результат и игнорирует битый json', () => {
    writeCachedScan(result('scan-a'))
    expect(readCachedScan('scan-a')?.wine?.title).toBe('Рислинг')
    expect(readCachedScan('missing')).toBeNull()
    sessionStorage.setItem(storageKeys.liveScans, '{')
    expect(readCachedScan('scan-a')).toBeNull()
  })

  it('хранит последние 12 результатов', () => {
    for (let index = 1; index <= 13; index += 1) {
      writeCachedScan(result(`scan-${index}`))
    }
    expect(readCachedScan('scan-1')).toBeNull()
    expect(readCachedScan('scan-2')?.scanId).toBe('scan-2')
    expect(readCachedScan('scan-13')?.scanId).toBe('scan-13')
  })
})
