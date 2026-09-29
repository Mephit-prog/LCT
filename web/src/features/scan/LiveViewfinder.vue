<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref, watch } from 'vue'
import UiButton from '@/components/ui/UiButton.vue'
import { useFocusTrap } from '@/composables/useFocusTrap'
import { logError } from '@/services/analytics/logger'
import { useScanStore } from './scan.store'
import { useScan } from './useScan'
import {
  bindVideoElement,
  captureStill,
  closeLiveCamera,
  markVideoReady,
  startLiveCamera,
  takeCapturedFile,
  useLiveCamera
} from './useLiveCamera'

const scanStore = useScanStore()
const { scanFile } = useScan()
const { phase, previewUrl, isReady, isOpen } = useLiveCamera()
const root = ref<HTMLElement | null>(null)
const videoRef = ref<HTMLVideoElement | null>(null)
const isCapturing = ref(false)

useFocusTrap(root, isOpen)

watch(videoRef, (element) => {
  bindVideoElement(element)
})

const onClose = (): void => {
  closeLiveCamera()
}

const onShutter = async (): Promise<void> => {
  if (isCapturing.value || !isReady.value) {
    return
  }
  isCapturing.value = true
  try {
    await captureStill()
  } catch (error) {
    logError('camera', error)
  } finally {
    isCapturing.value = false
  }
}

const onRetake = (): void => {
  const started = startLiveCamera()
  if (!started) {
    closeLiveCamera()
  }
}

const onSend = (): void => {
  const file = takeCapturedFile()
  if (file === null) {
    return
  }
  void scanFile(file, 'camera')
}

const onKeydown = (event: KeyboardEvent): void => {
  if (event.key !== 'Escape' || !isOpen.value) {
    return
  }
  closeLiveCamera()
}

onMounted(() => {
  document.addEventListener('keydown', onKeydown)
})

onBeforeUnmount(() => {
  document.removeEventListener('keydown', onKeydown)
})

watch(() => scanStore.isProcessing, (isProcessing) => {
  if (isProcessing && phase.value !== 'closed') {
    closeLiveCamera()
  }
})
</script>

<template>
  <div
    v-if="isOpen && !scanStore.isProcessing"
    ref="root"
    class="viewfinder"
    role="dialog"
    aria-modal="true"
    aria-label="Съёмка этикетки"
  >
    <div v-if="phase === 'live'" class="viewfinder__stage">
      <video
        ref="videoRef"
        class="viewfinder__video"
        autoplay
        playsinline
        muted
        @loadedmetadata="markVideoReady"
      />
      <div class="viewfinder__shade" aria-hidden="true">
        <div class="viewfinder__frame" />
      </div>
      <p class="viewfinder__hint">Этикетка целиком в рамке</p>
      <button type="button" class="viewfinder__close" aria-label="Закрыть камеру" @click="onClose">
        ×
      </button>
      <div class="viewfinder__shutter-wrap">
        <button
          type="button"
          class="viewfinder__shutter"
          aria-label="Сфотографировать"
          :disabled="!isReady || isCapturing"
          @click="onShutter"
        />
      </div>
    </div>

    <div v-else class="viewfinder__preview">
      <img
        v-if="previewUrl !== null"
        class="viewfinder__preview-image"
        :src="previewUrl"
        alt="Снимок этикетки"
      >
      <button type="button" class="viewfinder__close" aria-label="Закрыть камеру" @click="onClose">
        ×
      </button>
      <div class="viewfinder__actions">
        <UiButton variant="secondary" size="lg" is-block @click="onRetake">Переснять</UiButton>
        <UiButton size="lg" is-block @click="onSend">Отправить</UiButton>
      </div>
    </div>
  </div>
</template>

<style scoped>
.viewfinder {
  position: fixed;
  inset: 0;
  z-index: 58;
  display: flex;
  flex-direction: column;
  background: #111;
  color: var(--color-white);
}

.viewfinder__stage,
.viewfinder__preview {
  position: relative;
  flex: 1;
  min-height: 0;
}

.viewfinder__video,
.viewfinder__preview-image {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
}

.viewfinder__video {
  object-fit: cover;
  background: #000;
}

.viewfinder__preview-image {
  object-fit: contain;
  background: #111;
}

.viewfinder__shade {
  position: absolute;
  inset: 0;
  z-index: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
  pointer-events: none;
}

.viewfinder__frame {
  width: min(72vw, 280px);
  aspect-ratio: 3 / 4;
  border: 2px solid var(--color-white);
  border-radius: var(--radius-md);
  box-shadow: 0 0 0 100vmax rgb(0 0 0 / 48%);
}

.viewfinder__hint {
  position: absolute;
  right: var(--space-4);
  bottom: calc(112px + var(--safe-bottom));
  left: var(--space-4);
  z-index: 2;
  margin: 0;
  color: var(--color-white);
  font-size: var(--text-sm);
  text-align: center;
  text-shadow: 0 1px 4px rgb(0 0 0 / 65%);
}

.viewfinder__close {
  position: absolute;
  top: calc(var(--space-3) + var(--safe-top));
  left: var(--space-3);
  z-index: 2;
  width: var(--tap-size);
  height: var(--tap-size);
  border: 0;
  border-radius: var(--radius-pill);
  background: rgb(0 0 0 / 45%);
  color: var(--color-white);
  font-size: 28px;
  line-height: 1;
}

.viewfinder__shutter-wrap {
  position: absolute;
  right: 0;
  bottom: calc(var(--space-6) + var(--safe-bottom));
  left: 0;
  z-index: 2;
  display: flex;
  justify-content: center;
}

.viewfinder__shutter {
  width: 68px;
  height: 68px;
  padding: 0;
  border: 0;
  border-radius: 50%;
  background: var(--color-white);
  box-shadow: 0 0 0 4px rgb(255 255 255 / 35%);
}

.viewfinder__shutter:disabled {
  opacity: 0.4;
}

.viewfinder__close:focus-visible,
.viewfinder__shutter:focus-visible {
  outline: 2px solid var(--color-white);
  outline-offset: 3px;
}

.viewfinder__actions {
  position: absolute;
  right: 0;
  bottom: 0;
  left: 0;
  z-index: 2;
  display: flex;
  gap: var(--space-3);
  padding: var(--space-4);
  padding-bottom: calc(var(--space-4) + var(--safe-bottom));
}

.viewfinder__actions > * {
  flex: 1;
}
</style>
