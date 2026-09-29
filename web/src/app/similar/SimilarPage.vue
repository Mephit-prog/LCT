<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { useRoute } from 'vue-router'
import AgeFooter from '@/components/layout/AgeFooter.vue'
import PageHeader from '@/components/layout/PageHeader.vue'
import StickyBar from '@/components/layout/StickyBar.vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiChip from '@/components/ui/UiChip.vue'
import UiSkeleton from '@/components/ui/UiSkeleton.vue'
import InlineError from '@/components/wine/InlineError.vue'
import ReasonChips from '@/components/wine/ReasonChips.vue'
import WineSummaryCard from '@/components/wine/WineSummaryCard.vue'
import { useDebugPanel } from '@/composables/useDebugPanel'
import DebugPanel from '@/features/scan/DebugPanel.vue'
import ScanFileInput from '@/features/scan/ScanFileInput.vue'
import { applyRecognizedFilters, buildRecognizedFilters, type RecognizedFilter } from '@/features/scan/recognized-filters'
import { useScan } from '@/features/scan/useScan'
import { requestScannerCamera } from '@/features/scan/useLiveCamera'
import { useScanContext } from '@/features/scan/useScanContext'
import { track } from '@/services/analytics/events'
import type { ScanSource, Suggestion } from '@/services/types/api-types'

const route = useRoute()
const { scanFile } = useScan()
const isDebugVisible = useDebugPanel()

const scanId = computed(() => String(route.params.scanId ?? ''))
const scanIdRef = computed((): string | null => scanId.value === '' ? null : scanId.value)
const isFromCard = computed(() => route.query.from === 'card')

const { scanResult, isLoading, error, previewUrl, reload } = useScanContext(scanIdRef)

const selectedFilterIds = ref<string[]>([])
const fileInput = ref<InstanceType<typeof ScanFileInput> | null>(null)

const title = computed(() => isFromCard.value ? 'Возможно, вы искали одно из этих вин' : 'Точного совпадения в каталоге нет')
const subtitle = computed(() => isFromCard.value ? 'Подборка по вашему фото и распознанным атрибутам' : 'Мы подобрали вина, максимально близкие к вашему')

const filters = computed((): RecognizedFilter[] => buildRecognizedFilters(scanResult.value?.recognized ?? null))
const activeFilters = computed(() => filters.value.filter((filter) => selectedFilterIds.value.includes(filter.id)))
const similar = computed((): Suggestion[] => applyRecognizedFilters(scanResult.value?.similar ?? [], activeFilters.value))
const alternatives = computed((): Suggestion[] => applyRecognizedFilters(scanResult.value?.alternatives ?? [], activeFilters.value))
const isFilteredEmpty = computed(() => activeFilters.value.length > 0 && similar.value.length === 0 && alternatives.value.length === 0)

watch(scanResult, (result) => {
  if (result === null) {
    return
  }
  track({ name: 'similar_screen_viewed', scanId: result.scanId, reason: isFromCard.value ? 'user_rejected' : 'not_found' })
})

const toggleFilter = (filter: RecognizedFilter): void => {
  if (selectedFilterIds.value.includes(filter.id)) {
    selectedFilterIds.value = selectedFilterIds.value.filter((id) => id !== filter.id)
    return
  }
  selectedFilterIds.value = [...selectedFilterIds.value, filter.id]
}

const resetFilters = (): void => {
  selectedFilterIds.value = []
}

const isFilterSelected = (filter: RecognizedFilter): boolean => selectedFilterIds.value.includes(filter.id)

const wineRoute = (slug: string): { name: string, params: { slug: string }, query: { scan: string } } => ({
  name: 'wine',
  params: { slug },
  query: { scan: scanId.value }
})

const onAlternativeOpen = (item: Suggestion): void => {
  track({ name: 'alternative_opened', scanId: scanId.value, slug: item.wine.slug, reasons: item.reasons })
}

const onRescan = (): void => {
  track({ name: 'rescan_clicked', from: 'similar' })
  requestScannerCamera(() => {
    fileInput.value?.openCamera()
  })
}

const onFileSelected = (file: File, source: ScanSource): void => {
  void scanFile(file, source)
}

const onReload = (): void => {
  void reload()
}
</script>

<template>
  <main class="similar-page page" :class="{ 'page--debug': isDebugVisible && scanResult !== null }">
    <PageHeader back-label="К сканеру" back-to="/scan" />

    <header class="similar-page__hero">
      <div class="similar-page__photo">
        <img v-if="previewUrl !== null" :src="previewUrl" alt="Ваше фото этикетки" width="72" height="72">
        <span v-else class="similar-page__photo-placeholder" aria-hidden="true">◎</span>
      </div>
      <div class="similar-page__hero-text">
        <h1 class="similar-page__title">{{ title }}</h1>
        <p class="similar-page__subtitle">{{ subtitle }}</p>
      </div>
    </header>

    <div v-if="isLoading" class="similar-page__skeleton" aria-busy="true">
      <UiSkeleton v-for="index in 3" :key="index" height="128px" radius="var(--radius-md)" />
    </div>

    <InlineError
      v-else-if="error !== null"
      :title="error.message"
      hint="Результат скана не удалось загрузить. Повторите или отсканируйте бутылку ещё раз."
      @retry="onReload"
    />

    <template v-else-if="scanResult !== null">
      <section v-if="filters.length > 0" class="similar-page__section" aria-labelledby="recognized-title">
        <h2 id="recognized-title" class="section-title">Что мы распознали</h2>
        <ul class="similar-page__filters">
          <li v-for="filter in filters" :key="filter.id">
            <UiChip :label="filter.label" is-interactive :is-selected="isFilterSelected(filter)" @click="toggleFilter(filter)" />
          </li>
        </ul>
        <p v-if="isFilteredEmpty" class="similar-page__filters-empty">
          Под выбранные фильтры ничего не подошло.
          <button class="similar-page__filters-reset" type="button" @click="resetFilters">Сбросить</button>
        </p>
      </section>

      <section v-if="similar.length > 0" class="similar-page__section" aria-labelledby="similar-list-title">
        <h2 id="similar-list-title" class="section-title">Похожие вина</h2>
        <ul class="similar-page__list">
          <li v-for="item in similar.slice(0, 5)" :key="item.wine.slug">
            <WineSummaryCard :wine="item.wine" :to="wineRoute(item.wine.slug)" variant="list">
              <ReasonChips :reasons="item.reasons" />
            </WineSummaryCard>
          </li>
        </ul>
      </section>

      <section v-if="alternatives.length > 0" class="similar-page__section" aria-labelledby="alternatives-title">
        <h2 id="alternatives-title" class="section-title">Похожие вина других виноделен</h2>
        <ul class="similar-page__list">
          <li v-for="item in alternatives.slice(0, 6)" :key="item.wine.slug">
            <WineSummaryCard :wine="item.wine" :to="wineRoute(item.wine.slug)" variant="list" @open="onAlternativeOpen(item)">
              <ReasonChips :reasons="item.reasons" />
            </WineSummaryCard>
          </li>
        </ul>
      </section>

      <p class="similar-page__section">Предложение вина и цифровой сомелье пока недоступны.</p>
    </template>

    <AgeFooter />

    <StickyBar>
      <UiButton size="lg" @click="onRescan">Сканировать ещё</UiButton>
    </StickyBar>

    <DebugPanel v-if="isDebugVisible && scanResult !== null" :result="scanResult" />
    <ScanFileInput ref="fileInput" @select="onFileSelected" />
  </main>
</template>

<style scoped>
.similar-page__hero {
  display: flex;
  gap: var(--space-4);
  align-items: center;
  padding: var(--space-4);
  background: var(--color-bg-soft);
}

.similar-page__photo {
  flex-shrink: 0;
  width: 72px;
  height: 72px;
  overflow: hidden;
  border-radius: var(--radius-md);
  background: var(--color-text);
}

.similar-page__photo img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.similar-page__photo-placeholder {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 100%;
  height: 100%;
  color: var(--color-bg-tint-deep);
  font-size: var(--text-xl);
}

.similar-page__title {
  font-size: var(--text-lg);
  color: var(--color-primary-dark);
}

.similar-page__subtitle {
  margin-top: var(--space-1);
  color: var(--color-text-muted);
  font-size: var(--text-sm);
}

.similar-page__skeleton {
  display: flex;
  flex-direction: column;
  gap: var(--space-3);
  padding: var(--space-4);
}

.similar-page__section {
  padding: var(--space-4);
}

.similar-page__filters {
  display: flex;
  flex-wrap: wrap;
  gap: var(--space-2);
}

.similar-page__filters-empty {
  margin-top: var(--space-3);
  color: var(--color-text-muted);
  font-size: var(--text-sm);
}

.similar-page__filters-reset {
  padding: 0;
  border: 0;
  background: transparent;
  color: var(--color-primary);
  font: inherit;
  font-weight: 600;
  cursor: pointer;
  min-height: var(--tap-size);
}

.similar-page__list {
  display: flex;
  flex-direction: column;
  gap: var(--space-3);
}

.similar-page__suggest {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  gap: var(--space-2);
}

.similar-page__suggest-text {
  color: var(--color-text-muted);
}
</style>
