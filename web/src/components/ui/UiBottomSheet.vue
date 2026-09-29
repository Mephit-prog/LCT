<script setup lang="ts">
import { computed, onBeforeUnmount, ref, watch } from 'vue'
import { useFocusTrap } from '@/composables/useFocusTrap'

interface UiBottomSheetProps {
  title?: string
  isModal?: boolean
  maxHeight?: string
  ariaLabel?: string
}

const { title, isModal = true, maxHeight = '85dvh', ariaLabel } = defineProps<UiBottomSheetProps>()

const isOpen = defineModel<boolean>({ default: false })

const emit = defineEmits<{ close: [] }>()

const sheetRef = ref<HTMLElement | null>(null)
const dragOffset = ref(0)
const isDragging = ref(false)
let dragStartY: number | null = null

const isTrapActive = computed(() => isOpen.value && isModal)
useFocusTrap(sheetRef, isTrapActive)

const titleId = `ui-sheet-title-${Math.random().toString(36).slice(2, 8)}`

const close = (): void => {
  isOpen.value = false
  emit('close')
}

const onKeydown = (event: KeyboardEvent): void => {
  if (event.key !== 'Escape') {
    return
  }
  event.stopPropagation()
  close()
}

const onTouchStart = (event: TouchEvent): void => {
  const touch = event.touches[0]
  if (touch === undefined) {
    return
  }
  dragStartY = touch.clientY
  isDragging.value = true
}

const onTouchMove = (event: TouchEvent): void => {
  const touch = event.touches[0]
  if (dragStartY === null || touch === undefined) {
    return
  }
  dragOffset.value = Math.max(0, touch.clientY - dragStartY)
}

const onTouchEnd = (): void => {
  const shouldClose = dragOffset.value > 80
  dragStartY = null
  isDragging.value = false
  dragOffset.value = 0
  if (shouldClose) {
    close()
  }
}

const sheetStyle = computed(() => ({
  maxHeight,
  transform: dragOffset.value > 0 ? `translateY(${dragOffset.value}px)` : undefined,
  transition: isDragging.value ? 'none' : undefined
}))

const lockScroll = (shouldLock: boolean): void => {
  if (!isModal) {
    return
  }
  document.body.style.overflow = shouldLock ? 'hidden' : ''
}

watch(isOpen, (value) => {
  lockScroll(value)
  if (value) {
    document.addEventListener('keydown', onKeydown)
    return
  }
  document.removeEventListener('keydown', onKeydown)
})

onBeforeUnmount(() => {
  document.removeEventListener('keydown', onKeydown)
  lockScroll(false)
})
</script>

<template>
  <Teleport to="body">
    <Transition name="sheet">
      <div v-if="isOpen" class="ui-sheet" :class="{ 'ui-sheet--modal': isModal }">
        <div v-if="isModal" class="ui-sheet__overlay" @click="close" />
        <section
          ref="sheetRef"
          class="ui-sheet__panel"
          role="dialog"
          :aria-modal="isModal || undefined"
          :aria-labelledby="title === undefined ? undefined : titleId"
          :aria-label="title === undefined ? ariaLabel : undefined"
          :style="sheetStyle"
          tabindex="-1"
        >
          <div
            class="ui-sheet__grip"
            @touchstart.passive="onTouchStart"
            @touchmove.passive="onTouchMove"
            @touchend="onTouchEnd"
            @touchcancel="onTouchEnd"
          >
            <span class="ui-sheet__handle" aria-hidden="true" />
          </div>
          <header v-if="title !== undefined" class="ui-sheet__header">
            <h2 :id="titleId" class="ui-sheet__title">{{ title }}</h2>
            <button class="ui-sheet__close" type="button" aria-label="Закрыть" @click="close">×</button>
          </header>
          <div class="ui-sheet__body">
            <slot />
          </div>
        </section>
      </div>
    </Transition>
  </Teleport>
</template>

<style scoped>
.ui-sheet {
  position: fixed;
  inset: 0;
  z-index: 50;
  display: flex;
  align-items: flex-end;
  justify-content: center;
  pointer-events: none;
}

.ui-sheet--modal {
  pointer-events: auto;
}

.ui-sheet__overlay {
  position: absolute;
  inset: 0;
  background: var(--color-overlay);
}

.ui-sheet__panel {
  position: relative;
  display: flex;
  flex-direction: column;
  width: 100%;
  max-width: var(--container-max);
  padding-bottom: var(--safe-bottom);
  border-radius: var(--radius-lg) var(--radius-lg) 0 0;
  background: var(--color-bg);
  box-shadow: var(--shadow-sheet);
  pointer-events: auto;
  transition: transform var(--duration-base) var(--ease-out);
  outline: none;
}

.ui-sheet__grip {
  display: flex;
  justify-content: center;
  padding: var(--space-3) 0 var(--space-2);
  touch-action: none;
  cursor: grab;
}

.ui-sheet__handle {
  width: 40px;
  height: 4px;
  border-radius: var(--radius-pill);
  background: var(--color-border);
}

.ui-sheet__header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: var(--space-3);
  padding: 0 var(--space-4) var(--space-3);
}

.ui-sheet__title {
  font-size: var(--text-lg);
}

.ui-sheet__close {
  flex-shrink: 0;
  width: var(--tap-size);
  height: var(--tap-size);
  border: 0;
  border-radius: 50%;
  background: transparent;
  color: var(--color-text-muted);
  font-size: var(--text-xl);
  line-height: 1;
  cursor: pointer;
}

.ui-sheet__body {
  overflow-y: auto;
  padding: 0 var(--space-4) var(--space-4);
}

.sheet-enter-active,
.sheet-leave-active {
  transition: opacity var(--duration-base) var(--ease-out);
}

.sheet-enter-active .ui-sheet__panel,
.sheet-leave-active .ui-sheet__panel {
  transition: transform var(--duration-base) var(--ease-out);
}

.sheet-enter-from,
.sheet-leave-to {
  opacity: 0;
}

.sheet-enter-from .ui-sheet__panel,
.sheet-leave-to .ui-sheet__panel {
  transform: translateY(100%);
}
</style>
