import type { ApiError, ApiErrorCode } from '@/services/apis/errors'
import { createApiError } from '@/services/apis/errors'
import { mapWineCard, toWineSummary } from '@/services/apis/map-wine'
import { errorMessages } from '@/services/constants/messages'
import type { RecognizeCandidate, RecognizeOut, ScanResult, Suggestion, WineCard } from '@/services/types/api-types'
import { scanResultSchema } from '@/services/types/schemas'

export type RecognizeOutcome =
  | { kind: 'result', result: ScanResult }
  | { kind: 'needsSearch', text: string, scanId: string, latencyMs: number | null }
  | { kind: 'error', error: ApiError }

const toUnit = (score: number): number => {
  return Math.round(score * 100) / 10000
}

const similarLimit = 5
const alternativesLimit = 6
const neighborLimit = 8
const sheetLimit = 3
const topLimit = 5

const text = (value: string | null | undefined): string => {
  return value?.trim() ?? ''
}

const norm = (value: string | null | undefined): string => {
  return text(value).toLocaleLowerCase('ru')
}

const errorOf = (code: ApiErrorCode, cause?: unknown): { kind: 'error', error: ApiError } => {
  return {
    kind: 'error',
    error: createApiError(code, errorMessages[code], { cause })
  }
}

const sortCandidates = (candidates: readonly RecognizeCandidate[]): RecognizeCandidate[] => {
  return [...candidates].sort((left, right) => left.rank - right.rank)
}

const cardOf = (candidate: RecognizeCandidate): WineCard => {
  if (candidate.wine != null) {
    return candidate.wine
  }
  return {
    slug: candidate.slug,
    name: candidate.slug,
    producer: ''
  }
}

const sharesGrape = (left: string, right: string): boolean => {
  const rightGrapes = new Set(right.split(/[,;]/).map((part) => norm(part)).filter((part) => part !== ''))
  if (rightGrapes.size === 0) {
    return false
  }
  return left.split(/[,;]/).some((part) => rightGrapes.has(norm(part)))
}

const reasonsBetween = (anchor: WineCard | null, item: WineCard): string[] => {
  if (anchor === null) {
    return []
  }
  const reasons: string[] = []
  const leftProducer = norm(anchor.producer)
  const rightProducer = norm(item.producer)
  if (leftProducer !== '' && leftProducer === rightProducer) {
    reasons.push('same_manufacturer')
  } else if (leftProducer !== '' || rightProducer !== '') {
    reasons.push('other_manufacturer')
  }
  if (sharesGrape(text(anchor.grape), text(item.grape))) {
    reasons.push('same_grape')
  }
  const leftCategory = norm(anchor.category)
  const rightCategory = norm(item.category)
  if (leftCategory !== '' && leftCategory === rightCategory) {
    reasons.push('same_category')
  }
  return reasons
}

const toSuggestion = (anchor: WineCard | null, candidate: RecognizeCandidate): Suggestion => {
  const card = cardOf(candidate)
  return {
    wine: toWineSummary(card),
    reasons: reasonsBetween(anchor, card)
  }
}

const splitSuggestions = (
  pool: readonly RecognizeCandidate[],
  anchor: WineCard | null
): { similar: Suggestion[], alternatives: Suggestion[] } => {
  const anchorProducer = norm(anchor?.producer)
  const same = anchorProducer === ''
    ? []
    : pool.filter((item) => norm(cardOf(item).producer) === anchorProducer)
  const similarSource = same.length > 0 ? same : pool
  const similarItems = similarSource.slice(0, similarLimit)
  const used = new Set(similarItems.map((item) => item.slug))
  const alternatives = pool.filter((item) => {
    if (used.has(item.slug)) {
      return false
    }
    const producer = norm(cardOf(item).producer)
    return anchorProducer === '' || producer !== anchorProducer
  }).slice(0, alternativesLimit)
  return {
    similar: similarItems.map((item) => toSuggestion(anchor, item)),
    alternatives: alternatives.map((item) => toSuggestion(anchor, item))
  }
}

const buildResult = (
  status: 'exact' | 'uncertain' | 'not_found',
  scanId: string,
  ordered: readonly RecognizeCandidate[],
  anchorSlug: string | null,
  latencyMs: number | null
): ScanResult => {
  const chosen = anchorSlug === null ? null : ordered.find((item) => item.slug === anchorSlug) ?? null
  const reasonAnchor = chosen ?? ordered[0] ?? null
  const reasonCard = reasonAnchor === null ? null : cardOf(reasonAnchor)
  const pool = status === 'not_found'
    ? ordered
    : ordered.filter((item) => item.slug !== anchorSlug)
  const lists = splitSuggestions(pool, reasonCard)
  const anchorCandidate = anchorSlug === null
    ? ordered[0] ?? null
    : ordered.find((item) => item.slug === anchorSlug) ?? null
  const topSlug = anchorSlug ?? anchorCandidate?.slug ?? null
  const confidence = anchorCandidate === null ? null : toUnit(anchorCandidate.score)
  const second = ordered.find((item) => item.slug !== (topSlug ?? anchorCandidate?.slug))
  const margin = anchorCandidate !== null && second !== undefined
    ? toUnit(anchorCandidate.score - second.score)
    : null
  const listed = ordered.slice(0, topLimit).map((item) => ({
    slug: item.slug,
    confidence: toUnit(item.score)
  }))
  const top5 = topSlug !== null && !listed.some((item) => item.slug === topSlug)
    ? [{ slug: topSlug, confidence: confidence ?? 0 }, ...listed].slice(0, topLimit)
    : listed
  const wine = status === 'not_found' || anchorSlug === null
    ? null
    : mapWineCard(
      chosen === null ? { slug: anchorSlug, name: anchorSlug, producer: '' } : cardOf(chosen),
      pool.slice(0, neighborLimit).map((item) => toWineSummary(cardOf(item)))
    )
  return {
    scanId,
    match: {
      status,
      top1: topSlug === null ? null : { slug: topSlug, confidence: confidence ?? 0 },
      top5,
      margin,
      f1Top1: null,
      f1Top5: null,
      latencyMs
    },
    wine,
    candidates: pool.slice(0, sheetLimit).map((item) => toWineSummary(cardOf(item))),
    similar: lists.similar,
    alternatives: lists.alternatives,
    recognized: null
  }
}

type FinishedRecognize = Exclude<RecognizeOutcome, { kind: 'needsSearch' }>

const finish = (result: ScanResult): FinishedRecognize => {
  const parsed = scanResultSchema.safeParse(result)
  if (!parsed.success) {
    return errorOf('INVALID_RESPONSE', parsed.error)
  }
  return { kind: 'result', result: parsed.data }
}

export const notFoundFromCandidates = (
  scanId: string,
  candidates: readonly RecognizeCandidate[],
  latencyMs: number | null
): ScanResult => {
  const outcome = finish(buildResult('not_found', scanId, sortCandidates(candidates), null, latencyMs))
  if (outcome.kind === 'error') {
    throw outcome.error
  }
  return outcome.result
}

export const mapRecognize = (payload: RecognizeOut, scanId: string): RecognizeOutcome => {
  const reasons = payload.reason_codes ?? []
  const latencyMs = payload.timings_ms?.total ?? null
  const ordered = sortCandidates(payload.candidates ?? [])

  if (payload.status === 'error') {
    if (reasons.includes('deadline')) {
      return errorOf('TIMEOUT')
    }
    return errorOf('SERVER_ERROR')
  }
  if (payload.status === 'unreadable' || payload.status === 'ambiguous_scene') {
    return errorOf('UNSUPPORTED_IMAGE')
  }
  if (payload.status === 'unknown') {
    if (ordered.length === 0) {
      const query = text(payload.ocr?.normalized_text)
      if (query !== '') {
        return { kind: 'needsSearch', text: query, scanId, latencyMs }
      }
    }
    return finish(buildResult('not_found', scanId, ordered, null, latencyMs))
  }
  if (payload.status !== 'accepted' && payload.status !== 'ambiguous') {
    return errorOf('SERVER_ERROR')
  }
  const slug = payload.status === 'accepted'
    ? payload.slug ?? ordered[0]?.slug ?? null
    : ordered[0]?.slug ?? payload.slug ?? null
  if (slug === null) {
    return errorOf('UNSUPPORTED_IMAGE')
  }
  const status = payload.status === 'accepted' ? 'exact' : 'uncertain'
  return finish(buildResult(status, scanId, ordered, slug, latencyMs))
}
