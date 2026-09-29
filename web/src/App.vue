<script setup lang="ts">
import { onErrorCaptured, ref } from 'vue'
import { useRouter } from 'vue-router'
import AppErrorScreen from '@/app/error/AppErrorScreen.vue'
import ProcessingOverlay from '@/app/scan/ProcessingOverlay.vue'
import ScanErrorScreen from '@/app/scan/ScanErrorScreen.vue'
import UiToastHost from '@/components/ui/UiToastHost.vue'
import LiveViewfinder from '@/features/scan/LiveViewfinder.vue'
import { useScanStore } from '@/features/scan/scan.store'
import { logError } from '@/services/analytics/logger'

const router = useRouter()
const scanStore = useScanStore()
const hasRenderError = ref(false)

onErrorCaptured((error, _instance, info) => {
  logError('render', error, { info })
  hasRenderError.value = true
  return false
})

const onRecover = async (): Promise<void> => {
  scanStore.reset()
  hasRenderError.value = false
  await router.push({ name: 'scan' })
}
</script>

<template>
  <AppErrorScreen v-if="hasRenderError" @recover="onRecover" />
  <RouterView v-else />
  <LiveViewfinder />
  <ProcessingOverlay v-if="scanStore.isProcessing" />
  <ScanErrorScreen v-else-if="scanStore.phase === 'error'" />
  <UiToastHost />
</template>
