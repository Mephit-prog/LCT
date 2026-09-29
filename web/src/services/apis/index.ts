import { errorMessages } from '@/services/constants/messages'
import type { ScanResult, SommelierEvent, SommelierRequest, SuggestRequest, Wine, WineCard } from '@/services/types/api-types'
import { createApiError } from './errors'
import { readCachedScan } from './scan-cache'

async function* emptySommelier (): AsyncGenerator<SommelierEvent> {
  return
}

export const apis = {
  scan: async (image: Blob, signal: AbortSignal, fileName = 'label.jpg'): Promise<ScanResult> => {
    const { recognizeImage } = await import('./live-scan')
    return recognizeImage(image, signal, fileName)
  },
  getScan: (scanId: string, _signal?: AbortSignal): Promise<ScanResult> => {
    const cached = readCachedScan(scanId)
    if (cached === null) {
      return Promise.reject(createApiError('NOT_FOUND', errorMessages.NOT_FOUND, { status: 404, scanId }))
    }
    return Promise.resolve(cached)
  },
  getWine: async (slug: string, signal?: AbortSignal): Promise<Wine> => {
    const { loadWine } = await import('./live-scan')
    return loadWine(slug, signal)
  },
  searchWines: async (text: string, signal: AbortSignal): Promise<WineCard[]> => {
    const { searchWineCards } = await import('./live-scan')
    return searchWineCards(text, signal)
  },
  sommelier: (_body: SommelierRequest, _signal: AbortSignal): AsyncGenerator<SommelierEvent> => {
    return emptySommelier()
  },
  suggest: async (_scanId: string, _body: SuggestRequest): Promise<void> => {
    return
  }
}

export type Apis = typeof apis
