<script setup lang="ts">
import type { ToastItem } from '@/composables/useToast'

interface UiToastProps {
  toast: ToastItem
}

const { toast } = defineProps<UiToastProps>()

const emit = defineEmits<{ dismiss: [id: number] }>()

const onAction = (): void => {
  toast.onAction?.()
  emit('dismiss', toast.id)
}

const onDismiss = (): void => emit('dismiss', toast.id)
</script>

<template>
  <div class="ui-toast" :class="`ui-toast--${toast.tone}`" role="status">
    <span class="ui-toast__message">{{ toast.message }}</span>
    <button v-if="toast.actionLabel !== null" class="ui-toast__action" type="button" @click="onAction">
      {{ toast.actionLabel }}
    </button>
    <button class="ui-toast__close" type="button" aria-label="Закрыть уведомление" @click="onDismiss">×</button>
  </div>
</template>

<style scoped>
.ui-toast {
  display: flex;
  align-items: center;
  gap: var(--space-2);
  min-height: var(--tap-size);
  padding: var(--space-2) var(--space-2) var(--space-2) var(--space-4);
  border-radius: var(--radius-md);
  background: var(--color-text);
  color: var(--color-white);
  font-size: var(--text-sm);
  box-shadow: var(--shadow-card);
}

.ui-toast--success {
  background: var(--color-success);
}

.ui-toast--error {
  background: var(--color-danger);
}

.ui-toast__message {
  flex: 1;
}

.ui-toast__action {
  min-height: 36px;
  padding: 0 var(--space-3);
  border: 0;
  border-radius: var(--radius-sm);
  background: rgb(255 255 255 / 16%);
  color: inherit;
  font-weight: 600;
  cursor: pointer;
}

.ui-toast__close {
  width: 36px;
  height: 36px;
  border: 0;
  border-radius: var(--radius-sm);
  background: transparent;
  color: inherit;
  font-size: var(--text-lg);
  line-height: 1;
  cursor: pointer;
}
</style>
