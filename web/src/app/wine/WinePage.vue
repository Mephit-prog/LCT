<script setup lang="ts">
import { computed, onBeforeUnmount, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import PageHeader from '@/components/layout/PageHeader.vue'
import StickyBar from '@/components/layout/StickyBar.vue'
import AgeFooter from '@/components/layout/AgeFooter.vue'
import UiButton from '@/components/ui/UiButton.vue'
import DishesRow from '@/components/wine/DishesRow.vue'
import InlineError from '@/components/wine/InlineError.vue'
import ScanContextBar from '@/components/wine/ScanContextBar.vue'
import WineAttributes from '@/components/wine/WineAttributes.vue'
import WineCardSkeleton from '@/components/wine/WineCardSkeleton.vue'
import WineDescription from '@/components/wine/WineDescription.vue'
import WineHero from '@/components/wine/WineHero.vue'
import WineSummaryCard from '@/components/wine/WineSummaryCard.vue'
import { useDebugPanel } from '@/composables/useDebugPanel'
import { useShare } from '@/composables/useShare'
import { useToast } from '@/composables/useToast'
import DebugPanel from '@/features/scan/DebugPanel.vue'
import ScanFileInput from '@/features/scan/ScanFileInput.vue'
import UncertainSheet from '@/features/scan/UncertainSheet.vue'
import { useScanStore } from '@/features/scan/scan.store'
import { isUncertainHandled, markUncertainHandled } from '@/features/scan/uncertain-storage'
import { useRecentScans } from '@/features/scan/useRecentScans'
import { useScan } from '@/features/scan/useScan'
import { requestScannerCamera } from '@/features/scan/useLiveCamera'
import { useScanContext } from '@/features/scan/useScanContext'
import { apis } from '@/services/apis'
import { isAbortError, toApiError, type ApiError } from '@/services/apis/errors'
import { track } from '@/services/analytics/events'
import { logError } from '@/services/analytics/logger'
import { errorMessages, uiMessages } from '@/services/constants/messages'
import { uncertainSheetDelayMs } from '@/services/constants/timings'
import type { ScanSource, Wine } from '@/services/types/api-types'

const route = useRoute()
const router = useRouter()
const toast = useToast()
const store = useScanStore()
const { scanFile } = useScan()
const { share } = useShare()
const { remember } = useRecentScans()
const isDebugVisible = useDebugPanel()

const slug = computed(() => String(route.params.slug ?? ''))
const scanId = computed((): string | null => {
  const raw = route.query.scan
  return typeof raw === 'string' && raw !== '' ? raw : null
})

const { scanResult, previewUrl } = useScanContext(scanId)

const wine = ref<Wine | null>(null)
const isLoading = ref(true)
const loadError = ref<ApiError | null>(null)
const isSheetOpen = ref(false)
const fileInput = ref<InstanceType<typeof ScanFileInput> | null>(null)

let wineController: AbortController | null = null
let sheetTimer: ReturnType<typeof setTimeout> | null = null

const hasScanContext = computed(() => scanId.value !== null)
const similarRoute = computed(() => scanId.value === null
  ? null
  : { name: 'similar', params: { scanId: scanId.value }, query: { from: 'card' } })

const clearSheetTimer = (): void => {
  if (sheetTimer !== null) {
    clearTimeout(sheetTimer)
    sheetTimer = null
  }
}

const redirectAfterNotFound = async (): Promise<void> => {
  if (scanId.value !== null) {
    await router.replace({ name: 'similar', params: { scanId: scanId.value } })
    return
  }
  toast.error(uiMessages.wineNotFound)
  await router.replace({ name: 'scan' })
}

const loadWine = async (): Promise<void> => {
  wineController?.abort()
  wineController = new AbortController()
  const ownController = wineController
  isLoading.value = true
  loadError.value = null
  isSheetOpen.value = false
  clearSheetTimer()

  const cachedResult = scanId.value === null ? null : store.getResult(scanId.value)
  if (cachedResult?.wine !== null && cachedResult?.wine !== undefined && cachedResult.wine.slug === slug.value) {
    wine.value = cachedResult.wine
    isLoading.value = false
    return
  }

  try {
    const loaded = await apis.getWine(slug.value, ownController.signal)
    if (ownController.signal.aborted) {
      return
    }
    wine.value = loaded
  } catch (caught) {
    if (isAbortError(caught)) {
      return
    }
    const apiError = toApiError(caught, errorMessages.SERVER_ERROR)
    logError('wine', apiError, { slug: slug.value, scanId: scanId.value })
    if (apiError.code === 'NOT_FOUND') {
      await redirectAfterNotFound()
      return
    }
    wine.value = null
    loadError.value = apiError
  } finally {
    if (!ownController.signal.aborted) {
      isLoading.value = false
    }
  }
}

const scheduleUncertainSheet = (): void => {
  clearSheetTimer()
  const result = scanResult.value
  if (result === null || wine.value === null || scanId.value === null) {
    return
  }
  const isTopMatch = result.wine?.slug === wine.value.slug
  if (result.match.status !== 'uncertain' || !isTopMatch || result.candidates.length === 0 || isUncertainHandled(result.scanId)) {
    return
  }
  const currentScanId = result.scanId
  sheetTimer = setTimeout(() => {
    isSheetOpen.value = true
    track({ name: 'uncertain_sheet_shown', scanId: currentScanId, slug: slug.value })
  }, uncertainSheetDelayMs)
}

watch([wine, scanResult], ([loadedWine, result]) => {
  if (loadedWine === null) {
    return
  }
  track({ name: 'card_viewed', slug: loadedWine.slug, fromScan: hasScanContext.value })
  if (result !== null) {
    remember({ slug: loadedWine.slug, scanId: result.scanId, title: loadedWine.title })
  }
  scheduleUncertainSheet()
})

watch(slug, () => {
  void loadWine()
}, { immediate: true })

const onConfirmUncertain = (): void => {
  if (scanId.value === null) {
    return
  }
  markUncertainHandled(scanId.value)
  track({ name: 'uncertain_confirmed', scanId: scanId.value, slug: slug.value })
}

const onSelectCandidate = (candidateSlug: string): void => {
  if (scanId.value === null) {
    return
  }
  markUncertainHandled(scanId.value)
  track({ name: 'uncertain_candidate_selected', scanId: scanId.value, slug: candidateSlug })
}

const onRescan = (): void => {
  track({ name: 'rescan_clicked', from: 'card' })
  requestScannerCamera(() => {
    fileInput.value?.openCamera()
  })
}

const onFileSelected = (file: File, source: ScanSource): void => {
  void scanFile(file, source)
}

const onShare = (): void => {
  if (wine.value === null) {
    return
  }
  const url = new URL(route.fullPath, location.origin).toString()
  void share({ title: wine.value.title, text: `${wine.value.title} — ${wine.value.manufacturer.name}`, url })
}

const onRetryLoad = (): void => {
  void loadWine()
}

const summaryRoute = (summarySlug: string): { name: string, params: { slug: string }, query: Record<string, string> } => ({
  name: 'wine',
  params: { slug: summarySlug },
  query: scanId.value === null ? {} : { scan: scanId.value }
})

onBeforeUnmount(() => {
  wineController?.abort()
  clearSheetTimer()
})
</script>

<template>
  <main class="wine-page page" :class="{ 'page--debug': isDebugVisible && scanResult !== null }">
    <PageHeader back-label="Свои вина" back-to="/scan" />

    <ScanContextBar v-if="hasScanContext && scanResult !== null" :preview-url="previewUrl" />

    <WineCardSkeleton v-if="isLoading" />

    <InlineError
      v-else-if="loadError !== null"
      :title="loadError.message"
      hint="Проверьте соединение и попробуйте ещё раз"
      @retry="onRetryLoad"
    />

    <template v-else-if="wine !== null">
      <WineHero :wine="wine" />
      <WineAttributes :wine="wine" />
      <DishesRow :dishes="wine.dishes" />
      <WineDescription :text="wine.description" />
      <p>Цифровой сомелье пока недоступен.</p>

      <div v-if="similarRoute !== null" class="wine-page__not-it">
        <RouterLink :to="similarRoute" class="wine-page__not-it-link">Не это вино?</RouterLink>
      </div>

      <section v-if="wine.similar.length > 0" class="wine-page__similar" aria-labelledby="similar-title">
        <h2 id="similar-title" class="section-title wine-page__similar-title">Похожие вина</h2>
        <ul class="h-scroll wine-page__similar-list">
          <li v-for="item in wine.similar" :key="item.slug" class="wine-page__similar-item">
            <WineSummaryCard :wine="item" :to="summaryRoute(item.slug)" />
          </li>
        </ul>
      </section>

      <AgeFooter />

      <UncertainSheet
        v-if="scanResult !== null && scanId !== null"
        v-model="isSheetOpen"
        :candidates="scanResult.candidates"
        :scan-id="scanId"
        @confirm="onConfirmUncertain"
        @select="onSelectCandidate"
      />
    </template>

    <StickyBar>
      <UiButton size="lg" @click="onRescan">Сканировать ещё</UiButton>
      <UiButton variant="secondary" size="lg" class="sticky-bar__aux" aria-label="Поделиться" :is-disabled="wine === null" @click="onShare">
        Поделиться
      </UiButton>
    </StickyBar>

    <DebugPanel v-if="isDebugVisible && scanResult !== null" :result="scanResult" />
    <ScanFileInput ref="fileInput" @select="onFileSelected" />
  </main>
</template>

<style scoped>
.wine-page__not-it {
  display: flex;
  justify-content: center;
  padding: 0 var(--space-4) var(--space-4);
}

.wine-page__not-it-link {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-height: var(--tap-size);
  padding: 0 var(--space-4);
  border-radius: var(--radius-md);
  background: var(--color-bg-tint);
  color: var(--color-primary);
  font-weight: 600;
}

.wine-page__similar {
  padding: var(--space-4) 0;
}

.wine-page__similar-title {
  padding: 0 var(--space-4);
}

.wine-page__similar-list {
  padding-left: var(--space-4);
  padding-right: var(--space-4);
}

.wine-page__similar-item {
  display: flex;
  flex: 0 0 auto;
}
</style>
