<script setup lang="ts">
import { useToast } from '@/composables/useToast'
import UiToast from './UiToast.vue'

const { toasts, dismiss } = useToast()
</script>

<template>
  <div class="ui-toast-host" aria-live="polite">
    <TransitionGroup name="toast">
      <UiToast v-for="toast in toasts" :key="toast.id" :toast="toast" @dismiss="dismiss" />
    </TransitionGroup>
  </div>
</template>

<style scoped>
.ui-toast-host {
  position: fixed;
  left: 0;
  right: 0;
  bottom: calc(var(--sticky-height) + var(--safe-bottom) + var(--space-2));
  z-index: 60;
  display: flex;
  flex-direction: column;
  gap: var(--space-2);
  width: 100%;
  max-width: var(--container-max);
  margin: 0 auto;
  padding: 0 var(--space-4);
  pointer-events: none;
}

.ui-toast-host > * {
  pointer-events: auto;
}

.toast-enter-active,
.toast-leave-active {
  transition: opacity var(--duration-base) var(--ease-out), transform var(--duration-base) var(--ease-out);
}

.toast-enter-from,
.toast-leave-to {
  opacity: 0;
  transform: translateY(12px);
}
</style>
