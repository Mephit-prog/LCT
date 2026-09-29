<script setup lang="ts">
import { nextTick, onMounted, ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import AgeFooter from '@/components/layout/AgeFooter.vue'
import UiButton from '@/components/ui/UiButton.vue'
import RecentScans from '@/features/scan/RecentScans.vue'
import ScanFileInput from '@/features/scan/ScanFileInput.vue'
import { useScan } from '@/features/scan/useScan'
import ScanTipIcon from '@/features/scan/ScanTipIcon.vue'
import type { ScanTipIconName } from '@/features/scan/ScanTipIcon.vue'
import { requestScannerCamera } from '@/features/scan/useLiveCamera'
import type { ScanSource } from '@/services/types/api-types'

const route = useRoute()
const router = useRouter()
const { scanFile } = useScan()

const fileInput = ref<InstanceType<typeof ScanFileInput> | null>(null)
const cameraButton = ref<HTMLElement | null>(null)
const galleryButton = ref<HTMLElement | null>(null)

const tips: { icon: ScanTipIconName, text: string }[] = [
  { icon: 'frame', text: 'Этикетка целиком в кадре' },
  { icon: 'light', text: 'Без сильных бликов и теней' },
  { icon: 'parallel', text: 'Телефон параллельно бутылке' }
]

const onCamera = (): void => {
  requestScannerCamera(() => {
    fileInput.value?.openCamera()
  })
}
const onGallery = (): void => fileInput.value?.openGallery()

const onSelect = (file: File, source: ScanSource): void => {
  void scanFile(file, source)
}

const focusButton = (element: HTMLElement | null): void => {
  const target = element?.querySelector('button') ?? element
  target?.focus()
}

const applySourceQuery = async (): Promise<void> => {
  const source = route.query.source
  if (source !== 'camera' && source !== 'gallery') {
    return
  }
  await nextTick()
  if (source === 'camera') {
    onCamera()
    focusButton(cameraButton.value)
  } else {
    onGallery()
    focusButton(galleryButton.value)
  }
  const rest = { ...route.query }
  delete rest.source
  await router.replace({ query: rest })
}

onMounted(() => {
  void applySourceQuery()
})
</script>

<template>
  <main class="scan-page page">
    <header class="scan-page__header">
      <p class="scan-page__eyebrow">Своё Вино · Сканер</p>
      <h1 class="scan-page__title">Найти своё вино</h1>
      <p class="scan-page__subtitle">Сфотографируйте этикетку российского вина или загрузите фото</p>
    </header>

    <section class="scan-page__actions">
      <div ref="cameraButton">
        <UiButton size="lg" is-block @click="onCamera">Сканировать</UiButton>
      </div>
      <div ref="galleryButton">
        <UiButton variant="secondary" size="lg" is-block @click="onGallery">Загрузить фото</UiButton>
      </div>
    </section>

    <section class="scan-page__tips" aria-label="Подсказки по съёмке">
      <ul class="scan-page__tips-list">
        <li v-for="tip in tips" :key="tip.text" class="scan-page__tip">
          <span class="scan-page__tip-icon" aria-hidden="true">
            <ScanTipIcon :name="tip.icon" />
          </span>
          <span>{{ tip.text }}</span>
        </li>
      </ul>
    </section>

    <RecentScans />

    <AgeFooter />
    <ScanFileInput ref="fileInput" @select="onSelect" />
  </main>
</template>

<style scoped>
.scan-page {
  display: flex;
  flex-direction: column;
  gap: var(--space-6);
  min-height: 100dvh;
  padding-bottom: 0;
}

.scan-page__header {
  display: flex;
  flex-direction: column;
  gap: var(--space-2);
  padding: var(--space-8) var(--space-4) 0;
  padding-top: calc(var(--space-8) + var(--safe-top));
}

.scan-page__eyebrow {
  color: var(--color-primary);
  font-size: var(--text-xs);
  font-weight: 600;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.scan-page__title {
  font-size: 32px;
  color: var(--color-primary-dark);
}

.scan-page__subtitle {
  color: var(--color-text-muted);
}

.scan-page__actions {
  display: flex;
  flex-direction: column;
  gap: var(--space-3);
  padding: 0 var(--space-4);
}

.scan-page__tips {
  margin: 0 var(--space-4);
  padding: var(--space-4);
  border-radius: var(--radius-md);
  background: var(--color-bg-soft);
}

.scan-page__tips-list {
  display: flex;
  flex-direction: column;
  gap: var(--space-3);
}

.scan-page__tip {
  display: flex;
  align-items: center;
  gap: var(--space-3);
  font-size: var(--text-sm);
}

.scan-page__tip-icon {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
  width: 40px;
  height: 40px;
  border-radius: 50%;
  background: var(--color-bg-tint-strong);
  color: var(--color-primary-dark);
}

.scan-page :deep(.age-footer) {
  margin-top: auto;
}
</style>
