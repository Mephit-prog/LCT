<script setup lang="ts">
import UiButton from '@/components/ui/UiButton.vue'

interface InlineErrorProps {
  title: string
  hint?: string
  actionLabel?: string
}

const { title, hint, actionLabel = 'Повторить' } = defineProps<InlineErrorProps>()

const emit = defineEmits<{ retry: [] }>()

const onRetry = (): void => emit('retry')
</script>

<template>
  <div class="inline-error" role="alert">
    <p class="inline-error__title">{{ title }}</p>
    <p v-if="hint !== undefined" class="inline-error__hint">{{ hint }}</p>
    <UiButton variant="secondary" @click="onRetry">{{ actionLabel }}</UiButton>
  </div>
</template>

<style scoped>
.inline-error {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  gap: var(--space-3);
  margin: var(--space-4);
  padding: var(--space-4);
  border-radius: var(--radius-md);
  background: var(--color-bg-tint);
}

.inline-error__title {
  font-weight: 600;
  color: var(--color-primary-dark);
}

.inline-error__hint {
  color: var(--color-text-muted);
  font-size: var(--text-sm);
}
</style>
