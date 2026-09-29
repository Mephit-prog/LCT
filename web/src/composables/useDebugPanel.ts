import { computed, type ComputedRef } from 'vue'
import { useRoute } from 'vue-router'
import { env } from '@/services/constants/env'

export const useDebugPanel = (): ComputedRef<boolean> => {
  const route = useRoute()
  return computed(() => env.isDebugPanelForced || route.query.debug === '1' || route.query.debug === 'true')
}
