import { computed, onScopeDispose, ref, watch, type ComputedRef, type Ref } from 'vue'
import { isAbortError, toApiError, type ApiError } from '@/services/apis/errors'
import { logError } from '@/services/analytics/logger'
import { errorMessages } from '@/services/constants/messages'
import type { ScanResult } from '@/services/types/api-types'
import { useScanStore } from './scan.store'

export interface UseScanContext {
  scanResult: Ref<ScanResult | null>
  isLoading: Ref<boolean>
  error: Ref<ApiError | null>
  previewUrl: ComputedRef<string | null>
  reload: () => Promise<void>
}

export const useScanContext = (scanId: Ref<string | null>): UseScanContext => {
  const store = useScanStore()
  const scanResult = ref<ScanResult | null>(null)
  const isLoading = ref(false)
  const error = ref<ApiError | null>(null)
  let controller: AbortController | null = null

  const previewUrl = computed(() => scanId.value === null ? null : store.getPreview(scanId.value))

  const load = async (): Promise<void> => {
    controller?.abort()
    if (scanId.value === null) {
      scanResult.value = null
      error.value = null
      return
    }
    controller = new AbortController()
    const ownController = controller
    isLoading.value = true
    error.value = null
    try {
      const result = await store.loadResult(scanId.value, ownController.signal)
      if (ownController.signal.aborted) {
        return
      }
      scanResult.value = result
    } catch (caught) {
      if (isAbortError(caught)) {
        return
      }
      const apiError = toApiError(caught, errorMessages.SERVER_ERROR)
      logError('scan-context', apiError, { scanId: scanId.value })
      scanResult.value = null
      error.value = apiError
    } finally {
      if (!ownController.signal.aborted) {
        isLoading.value = false
      }
    }
  }

  watch(scanId, () => {
    void load()
  }, { immediate: true })

  onScopeDispose(() => controller?.abort())

  return { scanResult, isLoading, error, previewUrl, reload: load }
}
