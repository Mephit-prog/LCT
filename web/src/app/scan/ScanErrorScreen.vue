<script setup lang="ts">
import { computed, ref } from 'vue'
import UiButton from '@/components/ui/UiButton.vue'
import ScanFileInput from '@/features/scan/ScanFileInput.vue'
import { useScan } from '@/features/scan/useScan'
import { requestScannerCamera } from '@/features/scan/useLiveCamera'
import { useScanStore } from '@/features/scan/scan.store'
import type { ApiErrorCode } from '@/services/apis/errors'
import { errorMessages } from '@/services/constants/messages'
import type { ScanSource } from '@/services/types/api-types'

const store = useScanStore()
const { retry, scanFile } = useScan()
const fileInput = ref<InstanceType<typeof ScanFileInput> | null>(null)

const titles: Partial<Record<ApiErrorCode, string>> = {
  TIMEOUT: 'Не удалось получить ответ',
  NETWORK_FAILED: 'Не удалось отправить фото',
  SERVER_ERROR: 'Сервис временно недоступен',
  INVALID_RESPONSE: 'Сервис временно недоступен',
  IMAGE_TOO_LARGE: 'Фото слишком большое',
  UNSUPPORTED_IMAGE: 'Не удалось распознать изображение'
}

const hints: Partial<Record<ApiErrorCode, string>> = {
  TIMEOUT: 'Поиск занял больше 8 секунд. Проверьте соединение и попробуйте снова.',
  NETWORK_FAILED: 'Похоже, сеть пропала во время запроса. Фото сохранено — можно повторить отправку.',
  SERVER_ERROR: 'Мы уже знаем о проблеме. Повторите через несколько секунд.',
  INVALID_RESPONSE: 'Мы уже знаем о проблеме. Повторите через несколько секунд.',
  IMAGE_TOO_LARGE: 'Даже после сжатия файл не принят сервером. Попробуйте снять этикетку ещё раз.',
  UNSUPPORTED_IMAGE: 'Попробуйте другое фото: этикетка целиком, без бликов, телефон параллельно бутылке.'
}

const code = computed((): ApiErrorCode => store.error?.code ?? 'UNKNOWN')
const title = computed(() => titles[code.value] ?? errorMessages[code.value])
const hint = computed(() => hints[code.value] ?? 'Попробуйте ещё раз.')
const canRetry = computed(() => store.hasRetryableImage && code.value !== 'UNSUPPORTED_IMAGE')

const onRetry = (): void => {
  void retry()
}

const onRetake = (): void => {
  requestScannerCamera(() => {
    fileInput.value?.openCamera()
  })
}

const onSelect = (file: File, source: ScanSource): void => {
  store.reset()
  void scanFile(file, source)
}

const onBack = (): void => store.reset()
</script>

<template>
  <div class="scan-error" role="alertdialog" aria-labelledby="scan-error-title">
    <div class="scan-error__card">
      <h1 id="scan-error-title" class="scan-error__title">{{ title }}</h1>
      <p class="scan-error__hint">{{ hint }}</p>
      <div class="scan-error__actions">
        <UiButton v-if="canRetry" size="lg" is-block @click="onRetry">Повторить</UiButton>
        <UiButton :variant="canRetry ? 'secondary' : 'primary'" size="lg" is-block @click="onRetake">Снять заново</UiButton>
        <UiButton variant="transparent" @click="onBack">К экрану сканера</UiButton>
      </div>
    </div>
    <ScanFileInput ref="fileInput" @select="onSelect" />
  </div>
</template>

<style scoped>
.scan-error {
  position: fixed;
  inset: 0;
  z-index: 55;
  display: flex;
  align-items: flex-end;
  justify-content: center;
  background: var(--color-overlay);
}

.scan-error__card {
  display: flex;
  flex-direction: column;
  gap: var(--space-3);
  width: 100%;
  max-width: var(--container-max);
  padding: var(--space-6) var(--space-4) calc(var(--space-6) + var(--safe-bottom));
  border-radius: var(--radius-lg) var(--radius-lg) 0 0;
  background: var(--color-bg);
}

.scan-error__title {
  font-size: var(--text-xl);
  color: var(--color-primary-dark);
}

.scan-error__hint {
  color: var(--color-text-muted);
}

.scan-error__actions {
  display: flex;
  flex-direction: column;
  gap: var(--space-2);
  margin-top: var(--space-2);
}
</style>
