import { describe, expect, it } from 'vitest'
import { isApiError } from '@/services/apis/errors'
import { failureFromPayload } from '@/services/apis/http-client'
import { mapRecognize, notFoundFromCandidates } from '@/services/apis/map-recognize'
import type { RecognizeCandidate, RecognizeOut, WineCard } from '@/services/types/api-types'
import { scanResultSchema } from '@/services/types/schemas'

const card = (slug: string, name: string, producer: string, grape = 'рислинг', category = 'Белое'): WineCard => ({
  slug,
  name,
  producer,
  grape,
  category,
  color: 'Белое',
  sweetness: 'suhoe',
  media_url: `/api/media/${slug}.webp`
})

const candidate = (rank: number, score: number, wine: WineCard): RecognizeCandidate => ({
  wine_id: wine.slug,
  slug: wine.slug,
  rank,
  score,
  fuzzy_score: Math.round(score),
  edit_distance: 1,
  normalized_distance: 0.1,
  match: wine.grape ?? '',
  wine
})

const payload = (overrides: Partial<RecognizeOut> = {}): RecognizeOut => ({
  status: 'accepted',
  slug: 'fanagoria',
  reason_codes: [],
  timings_ms: { total: 84 },
  candidates: [
    candidate(1, 94.26, card('fanagoria', 'Рислинг', 'Фанагория')),
    candidate(2, 80, card('fanagoria-2', 'Рислинг Резерв', 'Фанагория', 'рислинг')),
    candidate(3, 40, card('other', 'Каберне', 'Лефкадия', 'каберне', 'Красное'))
  ],
  ...overrides
})

describe('mapRecognize', () => {
  it('accepted открывает карточку top-1 и кладёт соседей в похожие', () => {
    const outcome = mapRecognize(payload(), 'scan-1')
    expect(outcome.kind).toBe('result')
    if (outcome.kind !== 'result') {
      return
    }
    expect(scanResultSchema.safeParse(outcome.result).success).toBe(true)
    expect(outcome.result.match.status).toBe('exact')
    expect(outcome.result.wine?.slug).toBe('fanagoria')
    expect(outcome.result.match.top1).toEqual({ slug: 'fanagoria', confidence: 0.9426 })
    expect(outcome.result.match.margin).toBeCloseTo(0.1426)
    expect(outcome.result.match.f1Top1).toBeNull()
    expect(outcome.result.match.f1Top5).toBeNull()
    expect(outcome.result.match.latencyMs).toBe(84)
    expect(outcome.result.candidates.map((item) => item.slug)).toEqual(['fanagoria-2', 'other'])
    expect(outcome.result.wine?.similar.map((item) => item.slug)).toEqual(['fanagoria-2', 'other'])
    expect(outcome.result.similar.map((item) => item.wine.slug)).toEqual(['fanagoria-2'])
    expect(outcome.result.similar[0]?.reasons).toEqual(['same_manufacturer', 'same_grape', 'same_category'])
    expect(outcome.result.alternatives.map((item) => item.wine.slug)).toEqual(['other'])
    expect(outcome.result.alternatives[0]?.reasons).toContain('other_manufacturer')
    expect(outcome.result.recognized).toBeNull()
  })

  it('ambiguous показывает шторку без top-1 и не больше трёх кандидатов', () => {
    const extra = candidate(4, 30, card('third', 'Третье', 'Мысхако', 'каберне', 'Красное'))
    const outcome = mapRecognize(payload({
      status: 'ambiguous',
      slug: null,
      candidates: [...(payload().candidates ?? []), extra]
    }), 'scan-2')
    expect(outcome.kind).toBe('result')
    if (outcome.kind !== 'result') {
      return
    }
    expect(outcome.result.match.status).toBe('uncertain')
    expect(outcome.result.wine?.slug).toBe('fanagoria')
    expect(outcome.result.candidates).toHaveLength(3)
    expect(outcome.result.candidates.some((item) => item.slug === 'fanagoria')).toBe(false)
  })

  it('unknown делит кандидатов на похожие и альтернативы', () => {
    const outcome = mapRecognize(payload({ status: 'unknown', slug: null }), 'scan-3')
    expect(outcome.kind).toBe('result')
    if (outcome.kind !== 'result') {
      return
    }
    expect(outcome.result.match.status).toBe('not_found')
    expect(outcome.result.wine).toBeNull()
    expect(outcome.result.similar.map((item) => item.wine.slug)).toEqual(['fanagoria', 'fanagoria-2'])
    expect(outcome.result.alternatives.map((item) => item.wine.slug)).toEqual(['other'])
  })

  it('unknown без кандидатов просит текстовый поиск, если OCR что-то прочитал', () => {
    const outcome = mapRecognize(payload({
      status: 'unknown',
      slug: null,
      candidates: [],
      ocr: { normalized_text: '  рислинг фанагория  ' }
    }), 'scan-4')
    expect(outcome).toEqual({
      kind: 'needsSearch',
      text: 'рислинг фанагория',
      scanId: 'scan-4',
      latencyMs: 84
    })
  })

  it('unknown без текста и кандидатов остаётся пустой подборкой', () => {
    const outcome = mapRecognize(payload({
      status: 'unknown',
      slug: null,
      candidates: [],
      ocr: { normalized_text: '   ' }
    }), 'scan-5')
    expect(outcome.kind).toBe('result')
    if (outcome.kind !== 'result') {
      return
    }
    expect(outcome.result.similar).toEqual([])
    expect(outcome.result.alternatives).toEqual([])
    expect(outcome.result.match.top1).toBeNull()
  })

  it('поиск после unknown собирает not_found из кандидатов', () => {
    const result = notFoundFromCandidates('scan-6', [
      candidate(1, 70, card('found', 'Найдено', 'Фанагория'))
    ], 120)
    expect(result.match.status).toBe('not_found')
    expect(result.similar).toHaveLength(1)
    expect(result.match.latencyMs).toBe(120)
    expect(scanResultSchema.safeParse(result).success).toBe(true)
  })

  it('unreadable и пустой accepted становятся ошибкой съёмки', () => {
    const unreadable = mapRecognize(payload({ status: 'unreadable', slug: null, candidates: [] }), 'scan-7')
    const scene = mapRecognize(payload({ status: 'ambiguous_scene', slug: null, candidates: [] }), 'scan-8')
    const empty = mapRecognize(payload({ status: 'accepted', slug: null, candidates: [] }), 'scan-9')
    for (const outcome of [unreadable, scene, empty]) {
      expect(outcome.kind).toBe('error')
      if (outcome.kind !== 'error') {
        continue
      }
      expect(outcome.error.code).toBe('UNSUPPORTED_IMAGE')
    }
  })

  it('ошибка пайплайна различает дедлайн и занятость', () => {
    const deadline = mapRecognize(payload({ status: 'error', reason_codes: ['deadline'], slug: null, candidates: [] }), 'scan-10')
    const busy = mapRecognize(payload({ status: 'error', reason_codes: ['busy'], slug: null, candidates: [] }), 'scan-11')
    expect(deadline.kind === 'error' && deadline.error.code).toBe('TIMEOUT')
    expect(busy.kind === 'error' && busy.error.code).toBe('SERVER_ERROR')
  })
})

describe('ошибки HTTP бэкенда', () => {
  it('413 с invalid_size — слишком большое фото', () => {
    const error = failureFromPayload(413, { status: 'error', slug: null, reason_codes: ['invalid_size'] })
    expect(isApiError(error)).toBe(true)
    expect(error.code).toBe('IMAGE_TOO_LARGE')
    expect(error.serverCode).toBe('invalid_size')
  })

  it('404 not_found и 503 busy сохраняют код клиента', () => {
    expect(failureFromPayload(404, { status: 'error', reason_codes: ['not_found'] }).code).toBe('NOT_FOUND')
    expect(failureFromPayload(503, { status: 'error', reason_codes: ['busy'] }).code).toBe('SERVER_ERROR')
  })

  it('читает тело ошибки с полем error', () => {
    const error = failureFromPayload(422, { error: { code: 'UNSUPPORTED_IMAGE', message: 'нет' } })
    expect(error.code).toBe('UNSUPPORTED_IMAGE')
    expect(error.serverCode).toBe('UNSUPPORTED_IMAGE')
  })
})
