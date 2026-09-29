import { useRoute, useRouter } from 'vue-router'
import { useToast } from '@/composables/useToast'
import { logError } from '@/services/analytics/logger'
import { errorMessages } from '@/services/constants/messages'
import type { ScanResult, ScanSource } from '@/services/types/api-types'
import { isImagePrepareError } from './image-processing'
import { resolveScreen, toRouteLocation } from './resolve-screen'
import { useScanStore } from './scan.store'

export interface UseScan {
  scanFile: (file: File, source: ScanSource) => Promise<void>
  retry: () => Promise<void>
  cancel: () => void
  reset: () => void
}

export const useScan = (): UseScan => {
  const store = useScanStore()
  const router = useRouter()
  const route = useRoute()
  const toast = useToast()

  const navigateTo = async (result: ScanResult): Promise<void> => {
    const location = toRouteLocation(resolveScreen(result))
    const debug = route.query.debug
    if (typeof location === 'object' && typeof debug === 'string') {
      await router.push({ ...location, query: { ...location.query, debug } })
      return
    }
    await router.push(location)
  }

  const handleOutcome = async (result: ScanResult | null): Promise<void> => {
    if (result !== null) {
      await navigateTo(result)
      return
    }
    if (store.error?.code === 'NETWORK_OFFLINE') {
      toast.error(errorMessages.NETWORK_OFFLINE, { actionLabel: 'Повторить', onAction: () => void retry() })
    }
  }

  const scanFile = async (file: File, source: ScanSource): Promise<void> => {
    let result: ScanResult | null
    try {
      result = await store.start(file, source)
    } catch (error) {
      if (isImagePrepareError(error)) {
        toast.error(error.message)
        return
      }
      logError('scan', error)
      toast.error(errorMessages.UNKNOWN)
      return
    }
    await handleOutcome(result)
  }

  const retry = async (): Promise<void> => {
    let result: ScanResult | null
    try {
      result = await store.retry()
    } catch (error) {
      if (isImagePrepareError(error)) {
        toast.error(error.message)
        return
      }
      logError('scan', error)
      toast.error(errorMessages.UNKNOWN)
      return
    }
    await handleOutcome(result)
  }

  return {
    scanFile,
    retry,
    cancel: store.cancel,
    reset: store.reset
  }
}
