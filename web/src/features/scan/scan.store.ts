import { computed, ref } from 'vue'
import { defineStore } from 'pinia'
import { apis } from '@/services/apis'
import { isApiError, toApiError, type ApiError } from '@/services/apis/errors'
import { errorMessages } from '@/services/constants/messages'
import { maxImageBytesBeforeCompression, retryImageMaxSide } from '@/services/constants/timings'
import { track } from '@/services/analytics/events'
import { logError } from '@/services/analytics/logger'
import type { ScanResult, ScanSource } from '@/services/types/api-types'
import { createImagePrepareError, isImagePrepareError, validateImageFile, type PreparedImage } from './image-processing'
import { useImagePreparer } from './useImagePreparer'

export type ScanPhase = 'idle' | 'preparing' | 'uploading' | 'error'

const previewsLimit = 3

const revoke = (url: string | null): void => {
  if (url === null) {
    return
  }
  try {
    URL.revokeObjectURL(url)
  } catch {
    return
  }
}

export const useScanStore = defineStore('scan', () => {
  const preparer = useImagePreparer()

  const phase = ref<ScanPhase>('idle')
  const source = ref<ScanSource>('camera')
  const previewUrl = ref<string | null>(null)
  const startedAt = ref<number | null>(null)
  const error = ref<ApiError | null>(null)
  const results = ref<Record<string, ScanResult>>({})
  const previews = ref<Record<string, string>>({})

  let controller: AbortController | null = null
  let originalFile: File | null = null
  let prepared: PreparedImage | null = null

  const isProcessing = computed(() => phase.value === 'preparing' || phase.value === 'uploading')
  const hasRetryableImage = computed(() => originalFile !== null)

  const rememberResult = (result: ScanResult): void => {
    results.value = { ...results.value, [result.scanId]: result }
  }

  const getResult = (scanId: string): ScanResult | null => {
    return results.value[scanId] ?? null
  }

  const loadResult = async (scanId: string, signal?: AbortSignal): Promise<ScanResult> => {
    const cached = getResult(scanId)
    if (cached !== null) {
      return cached
    }
    const result = await apis.getScan(scanId, signal)
    rememberResult(result)
    return result
  }

  const getPreview = (scanId: string): string | null => {
    return previews.value[scanId] ?? null
  }

  const discardPreview = (): void => {
    revoke(previewUrl.value)
    previewUrl.value = null
  }

  const attachPreview = (scanId: string): void => {
    if (previewUrl.value === null) {
      return
    }
    const entries = Object.entries(previews.value)
    const excess = entries.slice(0, Math.max(0, entries.length + 1 - previewsLimit))
    for (const [key, url] of excess) {
      revoke(url)
      delete previews.value[key]
    }
    previews.value = { ...previews.value, [scanId]: previewUrl.value }
    previewUrl.value = null
  }

  const upload = async (image: PreparedImage, signal: AbortSignal): Promise<ScanResult> => {
    try {
      return await apis.scan(image.blob, signal)
    } catch (uploadError) {
      const isTooLarge = isApiError(uploadError) && uploadError.code === 'IMAGE_TOO_LARGE'
      const canShrink = originalFile !== null && Math.max(image.width, image.height) > retryImageMaxSide
      if (!isTooLarge || !canShrink || originalFile === null) {
        throw uploadError
      }
      prepared = await preparer.prepare(originalFile, { maxSide: retryImageMaxSide })
      return apis.scan(prepared.blob, signal)
    }
  }

  const run = async (): Promise<ScanResult | null> => {
    if (originalFile === null) {
      return null
    }
    controller?.abort()
    controller = new AbortController()
    error.value = null
    startedAt.value = Date.now()

    try {
      if (prepared === null) {
        phase.value = 'preparing'
        prepared = await preparer.prepare(originalFile)
        track({
          name: 'scan_image_prepared',
          originalBytes: prepared.originalBytes,
          sentBytes: prepared.sentBytes,
          resizeMs: prepared.resizeMs
        })
      }
      phase.value = 'uploading'
      const result = await upload(prepared, controller.signal)
      rememberResult(result)
      attachPreview(result.scanId)
      track({
        name: 'scan_completed',
        scanId: result.scanId,
        status: result.match.status,
        latencyMs: result.match.latencyMs,
        margin: result.match.margin
      })
      phase.value = 'idle'
      return result
    } catch (caught) {
      if (isImagePrepareError(caught)) {
        phase.value = 'idle'
        discardPreview()
        originalFile = null
        throw caught
      }
      const apiError = toApiError(caught, errorMessages.NETWORK_FAILED)
      if (apiError.code === 'ABORTED') {
        phase.value = 'idle'
        discardPreview()
        return null
      }
      logError('scan', apiError)
      track({ name: 'scan_failed', code: apiError.code })
      error.value = apiError
      phase.value = apiError.code === 'NETWORK_OFFLINE' ? 'idle' : 'error'
      return null
    }
  }

  const start = async (file: File, scanSource: ScanSource): Promise<ScanResult | null> => {
    if (isProcessing.value) {
      return null
    }
    const validation = validateImageFile(file, maxImageBytesBeforeCompression)
    if (validation !== null) {
      throw createImagePrepareError(validation)
    }
    discardPreview()
    previewUrl.value = URL.createObjectURL(file)
    originalFile = file
    prepared = null
    source.value = scanSource
    error.value = null
    track({ name: 'scan_started', source: scanSource })
    return run()
  }

  const retry = (): Promise<ScanResult | null> => {
    if (isProcessing.value) {
      return Promise.resolve(null)
    }
    return run()
  }

  const cancel = (): void => {
    controller?.abort()
    controller = null
    phase.value = 'idle'
    error.value = null
    discardPreview()
  }

  const reset = (): void => {
    cancel()
    originalFile = null
    prepared = null
  }

  return {
    phase,
    source,
    previewUrl,
    startedAt,
    error,
    isProcessing,
    hasRetryableImage,
    rememberResult,
    getResult,
    loadResult,
    getPreview,
    start,
    retry,
    cancel,
    reset
  }
})
