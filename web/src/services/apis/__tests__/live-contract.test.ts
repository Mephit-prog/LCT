// @vitest-environment node
/**
 * Живой контракт фронтенда с настоящим бэкендом. В обычном `npm run test` пропускается.
 *
 *   LIVE_API_URL=http://127.0.0.1:8080 npx vitest run live-contract
 *
 * LIVE_API_OCR_URL — второй бэкенд с WINE_MOCK_OCR и WINE_ROI_MODE=center_80_crop:
 * путь распознавания с кандидатами без платного OCR.
 * Ответы проходят те же zod-схемы и маппинги, что и в приложении.
 */
import { describe, expect, it } from 'vitest'
import { mapRecognize } from '@/services/apis/map-recognize'
import { mapWineCard, toWineSummary } from '@/services/apis/map-wine'
import type { WineCard } from '@/services/types/api-types'
import {
  backendErrorBodySchema,
  recognizeOutSchema,
  searchOutSchema,
  wineCardSchema,
  wineSchema,
  wineSummarySchema
} from '@/services/types/schemas'

const processEnv = (globalThis as unknown as { process?: { env: Record<string, string | undefined> } }).process?.env ?? {}
const apiUrl = processEnv.LIVE_API_URL ?? ''
const ocrApiUrl = processEnv.LIVE_API_OCR_URL ?? ''
const pageSize = 200
const heavy = 60_000
// 96×64 PNG: mock-OCR ignores pixels, and a small frame stays within the
// backend crop limits the way a client-normalized photo (≤1280 px) does.
const smallPhotoBase64 = 'iVBORw0KGgoAAAANSUhEUgAAAGAAAABACAIAAABqVuVZAAAAjklEQVR42u3buw2AMBBEQS+i/5aPCkh8wsjWvJyA0ULAJ1U19N6FABAgQIAAAQIESIAAAQIECNBZ3Z2Dk+xyntOPvSwI0I+XWH/AX9e/CVgQIECAAAECBAiQAAECBAgQIECABAgQIECAAAECJEBjyavnjT7zsCBAa4sf6iwIECBAgAABAiRAgAABAgTorB59wAyCLgV+sgAAAABJRU5ErkJggg=='

interface JsonResponse {
  status: number
  body: unknown
}

interface Health {
  ready: boolean
  recognition_ready: boolean
  wine_count: number
  catalog_sha256: string
  media_configured: boolean
}

interface Catalog {
  cards: WineCard[]
  invalid: string[]
}

const request = async (base: string, path: string, init?: RequestInit): Promise<JsonResponse> => {
  const response = await fetch(`${base}/api${path}`, init)
  return { status: response.status, body: await response.json() }
}

const health = async (base: string): Promise<Health> => {
  const response = await request(base, '/health')
  expect(response.status).toBe(200)
  return response.body as Health
}

const loadCatalog = async (base: string): Promise<Catalog> => {
  const cards: WineCard[] = []
  const invalid: string[] = []
  for (let offset = 0; ; offset += pageSize) {
    const page = await request(base, `/wines?limit=${pageSize}&offset=${offset}`)
    expect(page.status).toBe(200)
    const { total, items } = page.body as { total: number, items: unknown[] }
    for (const item of items) {
      const parsed = wineCardSchema.safeParse(item)
      if (parsed.success) {
        cards.push(parsed.data)
      } else {
        invalid.push(JSON.stringify(item).slice(0, 160))
      }
    }
    if (offset + pageSize >= total) {
      return { cards, invalid }
    }
  }
}

let catalog: Promise<Catalog> | null = null
const catalogOnce = (): Promise<Catalog> => {
  catalog ??= loadCatalog(apiUrl)
  return catalog
}

const hasMedia = (card: WineCard): boolean => (card.media_url ?? '') !== ''

const photoOf = async (base: string, card: WineCard): Promise<Blob> => {
  const response = await fetch(new URL(card.media_url ?? '', base))
  expect(response.status).toBe(200)
  return new Blob([await response.arrayBuffer()], { type: response.headers.get('content-type') ?? '' })
}

const recognize = async (base: string, image: Blob): Promise<JsonResponse> => {
  const form = new FormData()
  form.append('image', image, 'photo.webp')
  return request(base, '/recognize', { method: 'POST', body: form, headers: { 'X-Client-Normalized': '1' } })
}

describe.skipIf(apiUrl === '')('живой контракт: основной бэкенд', () => {
  it('health: каталог загружен, медиа подключены', async () => {
    const body = await health(apiUrl)
    expect(body.ready).toBe(true)
    expect(body.wine_count).toBeGreaterThan(0)
    expect(body.catalog_sha256).toMatch(/^[0-9a-f]{64}$/)
    expect(body.media_configured).toBe(true)
  })

  it('каталог: каждая карточка проходит схему и маппинги фронтенда', async () => {
    const { wine_count: wineCount } = await health(apiUrl)
    const { cards, invalid } = await catalogOnce()
    expect(invalid).toEqual([])
    expect(cards).toHaveLength(wineCount)
    const broken = cards
      .filter((card) => !wineSchema.safeParse(mapWineCard(card)).success
        || !wineSummarySchema.safeParse(toWineSummary(card)).success)
      .map((card) => card.slug)
    expect(broken).toEqual([])
    expect(cards.filter(hasMedia).length).toBeGreaterThan(0)
  }, heavy)

  it('карточка по slug совпадает со списком, 404 в общем формате ошибок', async () => {
    const { cards } = await catalogOnce()
    const first = cards[0]
    expect(first).toBeDefined()
    if (first === undefined) {
      return
    }
    const detail = await request(apiUrl, `/wines/${encodeURIComponent(first.slug)}`)
    expect(detail.status).toBe(200)
    expect(wineCardSchema.parse(detail.body)).toEqual(first)
    const missing = await request(apiUrl, '/wines/no-such-wine-live-contract')
    expect(missing.status).toBe(404)
    expect(backendErrorBodySchema.parse(missing.body).reason_codes).toEqual(['not_found'])
  }, heavy)

  it('медиа: фото карточек отдаются как изображения', async () => {
    const { cards } = await catalogOnce()
    const sample = cards.filter(hasMedia).slice(0, 20)
    expect(sample.length).toBeGreaterThan(0)
    for (const card of sample) {
      const response = await fetch(new URL(card.media_url ?? '', apiUrl))
      expect(response.status).toBe(200)
      expect(response.headers.get('content-type') ?? '').toMatch(/^image\//)
      await response.arrayBuffer()
    }
  }, heavy)

  it('поиск: ответ проходит searchOutSchema и находит производителя', async () => {
    const { cards } = await catalogOnce()
    const sample = cards.filter((_, index) => index % 400 === 0)
    for (const card of sample) {
      const response = await request(apiUrl, '/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: `${card.producer} ${card.name}`, k: 11, vintage_mode: 'soft' })
      })
      expect(response.status).toBe(200)
      const parsed = searchOutSchema.parse(response.body)
      expect(parsed.candidates.length).toBeGreaterThan(0)
      expect(parsed.candidates.every((candidate) => candidate.wine != null)).toBe(true)
      expect(parsed.candidates.some((candidate) => candidate.wine?.producer === card.producer)).toBe(true)
    }
  }, heavy)

  it('распознавание: ответ проходит схему, экран согласован с health', async () => {
    const status = await health(apiUrl)
    const { cards } = await catalogOnce()
    const card = cards.find(hasMedia)
    expect(card).toBeDefined()
    if (card === undefined) {
      return
    }
    const response = await recognize(apiUrl, await photoOf(apiUrl, card))
    expect(response.status).toBe(200)
    const payload = recognizeOutSchema.parse(response.body)
    const versions = (response.body as { versions?: { csv_sha256?: string } }).versions
    expect(versions?.csv_sha256).toBe(status.catalog_sha256)
    const outcome = mapRecognize(payload, 'live-contract-scan')
    if (status.recognition_ready) {
      expect(outcome.kind).not.toBe('error')
      return
    }
    // Без OCR или ROI бэкенд честно отказывается, фронт показывает экран пересъёмки или ошибку.
    expect(payload.reason_codes ?? []).not.toEqual([])
    expect(outcome.kind).toBe('error')
    if (outcome.kind === 'error') {
      expect(['UNSUPPORTED_IMAGE', 'SERVER_ERROR']).toContain(outcome.error.code)
    }
  }, heavy)
})

describe.skipIf(ocrApiUrl === '')('живой контракт: распознавание с кандидатами (mock-OCR)', () => {
  it('ambiguous с кандидатами → экран «не уверены» с карточкой', async () => {
    const status = await health(ocrApiUrl)
    expect(status.recognition_ready).toBe(true)
    const photo = new Blob([Uint8Array.from(atob(smallPhotoBase64), (char) => char.charCodeAt(0))], { type: 'image/png' })
    const response = await recognize(ocrApiUrl, photo)
    expect(response.status).toBe(200)
    const payload = recognizeOutSchema.parse(response.body)
    expect(payload.status).toBe('ambiguous')
    expect((payload.candidates ?? []).length).toBeGreaterThan(0)
    const outcome = mapRecognize(payload, 'live-contract-ocr')
    expect(outcome.kind).toBe('result')
    if (outcome.kind === 'result') {
      expect(outcome.result.match.status).toBe('uncertain')
      expect(outcome.result.wine).not.toBeNull()
      expect(outcome.result.candidates.length).toBeGreaterThan(0)
    }
  }, heavy)
})
