import type { z } from 'zod'
import type {
  matchCandidateSchema,
  matchSchema,
  matchStatusSchema,
  mediaSchema,
  recognizedSchema,
  scanResultSchema,
  sommelierModeSchema,
  sommelierRecommendationSchema,
  suggestionSchema,
  wineCardSchema,
  wineSchema,
  wineSummarySchema,
  recognizeCandidateSchema,
  recognizeOutSchema
} from './schemas'

export type Media = z.infer<typeof mediaSchema>
export type WineSummary = z.infer<typeof wineSummarySchema>
export type Wine = z.infer<typeof wineSchema>
export type WineCard = z.infer<typeof wineCardSchema>
export type RecognizeCandidate = z.infer<typeof recognizeCandidateSchema>
export type RecognizeOut = z.infer<typeof recognizeOutSchema>
export type MatchStatus = z.infer<typeof matchStatusSchema>
export type MatchCandidate = z.infer<typeof matchCandidateSchema>
export type Match = z.infer<typeof matchSchema>
export type Suggestion = z.infer<typeof suggestionSchema>
export type Recognized = z.infer<typeof recognizedSchema>
export type ScanResult = z.infer<typeof scanResultSchema>
export type SommelierMode = z.infer<typeof sommelierModeSchema>
export type SommelierRecommendation = z.infer<typeof sommelierRecommendationSchema>

export type ScanSource = 'camera' | 'gallery'

export interface SommelierAnswers {
  occasion: string
  dish: string | null
}

export interface SommelierRequest {
  mode: SommelierMode
  scanId: string | null
  slug: string | null
  answers: SommelierAnswers
}

export type SommelierEvent =
  | { type: 'token', text: string }
  | { type: 'recommendation', recommendation: SommelierRecommendation }
  | { type: 'done' }

export interface SuggestRequest {
  comment: string | null
}
