<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import UiButton from '@/components/ui/UiButton.vue'
import { useScanStore } from '@/features/scan/scan.store'
import { processingPhases } from '@/services/constants/timings'

const store = useScanStore()
const elapsedMs = ref(0)
let timer: ReturnType<typeof setInterval> | null = null

const tick = (): void => {
  if (store.startedAt === null) {
    elapsedMs.value = 0
    return
  }
  elapsedMs.value = Date.now() - store.startedAt
}

onMounted(() => {
  tick()
  timer = setInterval(tick, 200)
})

onBeforeUnmount(() => {
  if (timer !== null) {
    clearInterval(timer)
  }
})

const statusLabel = computed((): string => {
  if (elapsedMs.value >= processingPhases.almostReadyFromMs) {
    return 'Почти готово'
  }
  if (elapsedMs.value >= processingPhases.searchingFromMs) {
    return 'Ищем в каталоге'
  }
  return 'Обрабатываем фото'
})

const isSlow = computed(() => elapsedMs.value >= processingPhases.slowNoticeFromMs)

const onCancel = (): void => store.cancel()
</script>

<template>
  <div class="processing" role="status" aria-live="polite">
    <div class="processing__backdrop">
      <img v-if="store.previewUrl !== null" class="processing__preview" :src="store.previewUrl" alt="">
    </div>
    <div class="processing__content">
      <div class="processing__frame" aria-hidden="true">
        <img v-if="store.previewUrl !== null" class="processing__thumb" :src="store.previewUrl" alt="">
      </div>
      <div class="processing__bar" aria-hidden="true">
        <span class="processing__bar-fill" />
      </div>
      <p class="processing__label">{{ statusLabel }}</p>
      <p v-if="isSlow" class="processing__slow">Это занимает больше времени, чем обычно</p>
      <UiButton variant="transparent" class="processing__cancel" @click="onCancel">Отмена</UiButton>
    </div>
  </div>
</template>

<style scoped>
.processing {
  position: fixed;
  inset: 0;
  z-index: 72;
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
  background: var(--color-text);
}

.processing__backdrop {
  position: absolute;
  inset: 0;
  opacity: 0.35;
}

.processing__preview {
  width: 100%;
  height: 100%;
  object-fit: cover;
  filter: blur(18px) saturate(0.8);
  transform: scale(1.1);
}

.processing__content {
  position: relative;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: var(--space-4);
  width: min(100%, 320px);
  padding: var(--space-4);
  padding-bottom: calc(var(--space-4) + var(--safe-bottom));
}

.processing__frame {
  position: relative;
  width: 200px;
  height: 260px;
  padding: 6px;
  border: 2px solid var(--color-primary);
  border-radius: var(--radius-lg);
  animation: processing-pulse 1.6s ease-in-out infinite;
}

.processing__thumb {
  width: 100%;
  height: 100%;
  border-radius: calc(var(--radius-lg) - 6px);
  object-fit: cover;
}

.processing__bar {
  width: 100%;
  height: 4px;
  overflow: hidden;
  border-radius: var(--radius-pill);
  background: rgb(255 255 255 / 16%);
}

.processing__bar-fill {
  display: block;
  width: 40%;
  height: 100%;
  border-radius: var(--radius-pill);
  background: var(--color-primary);
  animation: processing-slide 1.4s var(--ease-out) infinite;
}

.processing__label {
  color: var(--color-white);
  font-size: var(--text-lg);
  font-family: var(--font-heading);
  text-align: center;
}

.processing__slow {
  color: var(--color-text-disabled);
  font-size: var(--text-sm);
  text-align: center;
}

.processing__cancel {
  color: var(--color-white);
  min-height: var(--tap-size);
}

@keyframes processing-slide {
  from {
    transform: translateX(-100%);
  }

  to {
    transform: translateX(250%);
  }
}

@keyframes processing-pulse {
  0%,
  100% {
    box-shadow: 0 0 0 0 rgb(143 61 66 / 60%);
  }

  50% {
    box-shadow: 0 0 0 12px rgb(143 61 66 / 0%);
  }
}
</style>
