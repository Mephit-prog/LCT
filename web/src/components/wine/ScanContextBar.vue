<script setup lang="ts">
import { onBeforeUnmount, ref, watch } from 'vue'

interface ScanContextBarProps {
  previewUrl: string | null
  text?: string
}

const { previewUrl, text = 'Найдено по вашему фото' } = defineProps<ScanContextBarProps>()

const isViewerOpen = ref(false)

const open = (): void => {
  if (previewUrl === null) {
    return
  }
  isViewerOpen.value = true
}

const close = (): void => {
  isViewerOpen.value = false
}

const onKeydown = (event: KeyboardEvent): void => {
  if (event.key === 'Escape') {
    close()
  }
}

watch(isViewerOpen, (value) => {
  if (value) {
    document.addEventListener('keydown', onKeydown)
    return
  }
  document.removeEventListener('keydown', onKeydown)
})

onBeforeUnmount(() => document.removeEventListener('keydown', onKeydown))
</script>

<template>
  <div class="scan-context">
    <button v-if="previewUrl !== null" class="scan-context__thumb" type="button" aria-label="Открыть ваше фото" @click="open">
      <img :src="previewUrl" alt="" width="40" height="40">
    </button>
    <span v-else class="scan-context__icon" aria-hidden="true">✓</span>
    <span class="scan-context__text">{{ text }}</span>
    <Teleport to="body">
      <div v-if="isViewerOpen && previewUrl !== null" class="scan-context__viewer" role="dialog" aria-label="Ваше фото" @click="close">
        <img class="scan-context__viewer-image" :src="previewUrl" alt="Ваше фото этикетки">
        <button class="scan-context__viewer-close" type="button" aria-label="Закрыть" @click="close">×</button>
      </div>
    </Teleport>
  </div>
</template>

<style scoped>
.scan-context {
  display: flex;
  align-items: center;
  gap: var(--space-3);
  min-height: var(--tap-size);
  padding: var(--space-2) var(--space-4);
  background: var(--color-bg-tint);
  color: var(--color-primary-dark);
  font-size: var(--text-sm);
  font-weight: 600;
}

.scan-context__thumb {
  flex-shrink: 0;
  width: 40px;
  height: 40px;
  padding: 0;
  border: 0;
  border-radius: var(--radius-sm);
  overflow: hidden;
  cursor: zoom-in;
  background: var(--color-text);
}

.scan-context__thumb img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.scan-context__icon {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 28px;
  height: 28px;
  border-radius: 50%;
  background: var(--color-primary);
  color: var(--color-white);
  font-size: var(--text-sm);
}

.scan-context__viewer {
  position: fixed;
  inset: 0;
  z-index: 70;
  display: flex;
  align-items: center;
  justify-content: center;
  background: rgb(44 42 40 / 96%);
}

.scan-context__viewer-image {
  max-width: 100%;
  max-height: 100%;
  object-fit: contain;
}

.scan-context__viewer-close {
  position: absolute;
  top: calc(var(--space-3) + var(--safe-top));
  right: var(--space-3);
  width: var(--tap-size);
  height: var(--tap-size);
  border: 0;
  border-radius: 50%;
  background: rgb(255 255 255 / 16%);
  color: var(--color-white);
  font-size: var(--text-xl);
  cursor: pointer;
}
</style>
