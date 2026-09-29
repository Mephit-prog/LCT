import type { MatchStatus, ScanSource, SommelierMode } from '@/services/types/api-types'

export type AnalyticsEvent =
  | { name: 'scan_started', source: ScanSource }
  | { name: 'scan_image_prepared', originalBytes: number, sentBytes: number, resizeMs: number }
  | { name: 'scan_completed', scanId: string, status: MatchStatus, latencyMs: number | null, margin: number | null }
  | { name: 'scan_failed', code: string }
  | { name: 'card_viewed', slug: string, fromScan: boolean }
  | { name: 'uncertain_sheet_shown', scanId: string, slug: string }
  | { name: 'uncertain_candidate_selected', scanId: string, slug: string }
  | { name: 'uncertain_confirmed', scanId: string, slug: string }
  | { name: 'similar_screen_viewed', scanId: string, reason: 'not_found' | 'user_rejected' }
  | { name: 'alternative_opened', scanId: string, slug: string, reasons: string[] }
  | { name: 'sommelier_started', mode: SommelierMode, slug: string | null }
  | { name: 'sommelier_answered', mode: SommelierMode, occasion: string, dish: string | null, slug: string | null }
  | { name: 'sommelier_recommendation_opened', mode: SommelierMode, slug: string }
  | { name: 'suggest_submitted', scanId: string }
  | { name: 'rescan_clicked', from: 'card' | 'similar' }

export const track = (event: AnalyticsEvent): void => {
  console.debug('[svoe-vino:analytics]', event)
}
