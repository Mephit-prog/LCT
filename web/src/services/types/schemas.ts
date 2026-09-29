import { z } from 'zod'

export const mediaSchema = z.object({
  url: z.string(),
  altText: z.string().nullable()
})

export const wineSummarySchema = z.object({
  slug: z.string().min(1),
  title: z.string().min(1),
  image: mediaSchema,
  manufacturer: z.string(),
  region: z.string(),
  category: z.string(),
  color: z.string().nullable(),
  publicRating: z.number().nullable()
})

export const wineSchema = z.object({
  slug: z.string().min(1),
  title: z.string().min(1),
  image: mediaSchema,
  manufacturer: z.object({ name: z.string(), slug: z.string() }),
  region: z.object({ name: z.string(), image: mediaSchema.nullable() }),
  grapes: z.array(z.object({ name: z.string(), image: mediaSchema.nullable() })),
  category: z.object({ name: z.string(), backgroundGradient: z.string().nullable() }),
  color: z.string().nullable(),
  alcohol: z.number().nullable(),
  temperature: z.string().nullable(),
  description: z.string().nullable(),
  dishes: z.array(z.object({ name: z.string(), image: mediaSchema.nullable() })),
  publicRating: z.number().nullable(),
  roskachestvoRating: z.object({ score: z.number(), year: z.number() }).nullable(),
  similar: z.array(wineSummarySchema)
})

export const matchStatusSchema = z.enum(['exact', 'uncertain', 'not_found'])

export const matchCandidateSchema = z.object({
  slug: z.string(),
  confidence: z.number()
})

export const matchSchema = z.object({
  status: matchStatusSchema,
  top1: matchCandidateSchema.nullable(),
  top5: z.array(matchCandidateSchema),
  margin: z.number().nullable(),
  f1Top1: z.number().nullable(),
  f1Top5: z.number().nullable(),
  latencyMs: z.number().nullable()
})

export const suggestionSchema = z.object({
  wine: wineSummarySchema,
  reasons: z.array(z.string())
})

export const recognizedSchema = z.object({
  manufacturer: z.string().nullable(),
  grapes: z.array(z.string()),
  category: z.string().nullable(),
  year: z.number().nullable(),
  region: z.string().nullable()
})

export const scanResultSchema = z.object({
  scanId: z.string().min(1),
  match: matchSchema,
  wine: wineSchema.nullable(),
  candidates: z.array(wineSummarySchema),
  similar: z.array(suggestionSchema),
  alternatives: z.array(suggestionSchema),
  recognized: recognizedSchema.nullable()
})

export const apiErrorBodySchema = z.object({
  error: z.object({
    code: z.string(),
    message: z.string().nullable().optional()
  })
})

export const backendErrorBodySchema = z.object({
  status: z.string(),
  reason_codes: z.array(z.string())
})

export const wineCardSchema = z.object({
  slug: z.string().min(1),
  name: z.string(),
  producer: z.string(),
  color: z.string().optional(),
  vintage: z.string().optional(),
  grape: z.string().optional(),
  category: z.string().optional(),
  sweetness: z.string().optional(),
  media_url: z.string().nullable().optional(),
  media_mapping_status: z.string().nullable().optional()
})

export const recognizeCandidateSchema = z.object({
  wine_id: z.string(),
  slug: z.string().min(1),
  rank: z.number(),
  score: z.number(),
  // Class-only candidates have no text/fuzzy branch yet: backend returns null.
  fuzzy_score: z.number().nullable(),
  edit_distance: z.number().nullable(),
  normalized_distance: z.number().nullable(),
  match: z.string().nullable(),
  wine: wineCardSchema.nullable().optional()
})

export const recognizeOutSchema = z.object({
  status: z.string(),
  slug: z.string().nullable().optional(),
  reason_codes: z.array(z.string()).optional(),
  timings_ms: z.object({
    total: z.number().optional()
  }).optional(),
  candidates: z.array(recognizeCandidateSchema).optional(),
  ocr: z.object({
    normalized_text: z.string().nullable().optional(),
    raw_text: z.string().nullable().optional(),
    status: z.string().optional()
  }).nullable().optional()
})

export const searchOutSchema = z.object({
  text: z.string(),
  backend: z.string(),
  candidates: z.array(recognizeCandidateSchema)
})

export const sommelierModeSchema = z.enum(['pairing', 'discovery'])

export const sommelierRecommendationSchema = z.object({
  wine: wineSummarySchema,
  reason: z.string()
})

export const sommelierTokenSchema = z.object({ text: z.string() })
