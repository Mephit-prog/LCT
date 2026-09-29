import { createApiError, isAbortError } from '@/services/apis/errors'
import { requestLive } from '@/services/apis/http-client'
import { mapRecognize, notFoundFromCandidates } from '@/services/apis/map-recognize'
import { mapWineCard } from '@/services/apis/map-wine'
import { writeCachedScan } from '@/services/apis/scan-cache'
import { logError } from '@/services/analytics/logger'
import { env } from '@/services/constants/env'
import { errorMessages } from '@/services/constants/messages'
import { liveScanTimeoutMs } from '@/services/constants/timings'
import type { RecognizeCandidate, RecognizeOut, ScanResult, Wine, WineCard } from '@/services/types/api-types'
import { recognizeOutSchema, searchOutSchema, wineCardSchema } from '@/services/types/schemas'

const searchLimit = 11

const liveTimeout = (): number => {
  return Math.max(env.scanTimeoutMs, liveScanTimeoutMs)
}

const fetchWineCard = async (slug: string, signal?: AbortSignal): Promise<WineCard> => {
  const response = await requestLive(`/wines/${encodeURIComponent(slug)}`, {
    signal,
    timeoutMs: env.scanTimeoutMs
  })
  const parsed = wineCardSchema.safeParse(response.data)
  if (!parsed.success) {
    logError('http', parsed.error, { path: `/wines/${slug}`, issues: parsed.error.issues })
    throw createApiError('INVALID_RESPONSE', errorMessages.INVALID_RESPONSE, { cause: parsed.error })
  }
  return parsed.data
}

const searchCandidates = async (text: string, signal: AbortSignal): Promise<RecognizeCandidate[]> => {
  try {
    const response = await requestLive('/search', {
      method: 'POST',
      data: { text, k: searchLimit, vintage_mode: 'soft' },
      headers: { 'Content-Type': 'application/json' },
      signal,
      timeoutMs: liveTimeout()
    })
    const parsed = searchOutSchema.safeParse(response.data)
    if (!parsed.success) {
      logError('http', parsed.error, { path: '/search', issues: parsed.error.issues })
      return []
    }
    return parsed.data.candidates
  } catch (error) {
    if (isAbortError(error)) {
      throw error
    }
    logError('http', error, { path: '/search' })
    return []
  }
}

const hydrateAnchor = async (payload: RecognizeOut, signal: AbortSignal): Promise<RecognizeOut> => {
  const candidates = payload.candidates ?? []
  const ordered = [...candidates].sort((left, right) => left.rank - right.rank)
  const slug = payload.status === 'accepted'
    ? payload.slug ?? ordered[0]?.slug ?? null
    : ordered[0]?.slug ?? null
  if (slug === null || payload.status === 'unknown' || payload.status === 'unreadable' || payload.status === 'error') {
    return payload
  }
  const index = candidates.findIndex((item) => item.slug === slug)
  const current = index === -1 ? null : candidates[index]
  if (current?.wine != null) {
    return payload
  }
  try {
    const card = await fetchWineCard(slug, signal)
    if (index === -1) {
      return {
        ...payload,
        candidates: [
          ...candidates,
          {
            wine_id: card.slug,
            slug: card.slug,
            rank: ordered.length + 1,
            score: 0,
            fuzzy_score: 0,
            edit_distance: 0,
            normalized_distance: 0,
            match: '',
            wine: card
          }
        ]
      }
    }
    const next = [...candidates]
    const target = next[index]
    if (target === undefined) {
      return payload
    }
    next[index] = { ...target, wine: card }
    return { ...payload, candidates: next }
  } catch (error) {
    if (isAbortError(error)) {
      throw error
    }
    logError('http', error, { path: `/wines/${slug}` })
    return payload
  }
}

export const searchWineCards = async (text: string, signal: AbortSignal): Promise<WineCard[]> => {
  const candidates = await searchCandidates(text, signal)
  const seen = new Set<string>()
  const result: WineCard[] = []
  for (const candidate of candidates) {
    if (candidate.wine == null || seen.has(candidate.slug)) {
      continue
    }
    seen.add(candidate.slug)
    result.push(candidate.wine)
  }
  return result
}

export const recognizeImage = async (image: Blob, signal: AbortSignal, fileName: string): Promise<ScanResult> => {
  const form = new FormData()
  form.append('image', image, fileName)
  const response = await requestLive('/recognize', {
    method: 'POST',
    data: form,
    headers: { 'X-Client-Normalized': '1' },
    signal,
    timeoutMs: liveTimeout()
  })
  const parsed = recognizeOutSchema.safeParse(response.data)
  if (!parsed.success) {
    logError('http', parsed.error, { path: '/recognize', issues: parsed.error.issues })
    throw createApiError('INVALID_RESPONSE', errorMessages.INVALID_RESPONSE, { cause: parsed.error })
  }
  const hydrated = await hydrateAnchor(parsed.data, signal)
  const scanId = crypto.randomUUID()
  const outcome = mapRecognize(hydrated, scanId)
  if (outcome.kind === 'error') {
    throw outcome.error
  }
  const result = outcome.kind === 'needsSearch'
    ? notFoundFromCandidates(outcome.scanId, await searchCandidates(outcome.text, signal), outcome.latencyMs)
    : outcome.result
  writeCachedScan(result)
  return result
}

export const loadWine = async (slug: string, signal?: AbortSignal): Promise<Wine> => {
  const card = await fetchWineCard(slug, signal)
  return mapWineCard(card)
}
