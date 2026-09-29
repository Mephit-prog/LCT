<script setup lang="ts">
interface UiChipProps {
  label: string
  isSelected?: boolean
  isInteractive?: boolean
  isDisabled?: boolean
  imageUrl?: string | null
  size?: 'sm' | 'md'
}

const {
  label,
  isSelected = false,
  isInteractive = false,
  isDisabled = false,
  imageUrl = null,
  size = 'md'
} = defineProps<UiChipProps>()

const emit = defineEmits<{ click: [] }>()

const onClick = (): void => {
  if (!isInteractive || isDisabled) {
    return
  }
  emit('click')
}
</script>

<template>
  <component
    :is="isInteractive ? 'button' : 'span'"
    class="ui-chip"
    :class="[`ui-chip--${size}`, { 'ui-chip--selected': isSelected, 'ui-chip--interactive': isInteractive, 'ui-chip--image': imageUrl !== null }]"
    :type="isInteractive ? 'button' : undefined"
    :aria-pressed="isInteractive ? isSelected : undefined"
    :disabled="isInteractive && isDisabled ? true : undefined"
    @click="onClick"
  >
    <img v-if="imageUrl !== null" class="ui-chip__image" :src="imageUrl" alt="" loading="lazy" decoding="async" width="28" height="28">
    <slot name="icon" />
    <span class="ui-chip__label">{{ label }}</span>
  </component>
</template>

<style scoped>
.ui-chip {
  display: inline-flex;
  align-items: center;
  gap: var(--space-2);
  max-width: 100%;
  padding: 0 var(--space-3);
  border: 1px solid var(--color-border-soft);
  border-radius: var(--radius-pill);
  background: var(--color-white);
  color: var(--color-text);
  font-size: var(--text-sm);
  line-height: 1.2;
  white-space: nowrap;
  -webkit-tap-highlight-color: transparent;
  transition: background-color var(--duration-fast) var(--ease-out), border-color var(--duration-fast) var(--ease-out);
}

.ui-chip--md {
  min-height: 36px;
}

.ui-chip--sm {
  min-height: 28px;
  padding: 0 var(--space-2);
  font-size: var(--text-xs);
  gap: var(--space-1);
}

.ui-chip--image {
  padding-left: var(--space-1);
}

.ui-chip--interactive {
  min-height: var(--tap-size);
  cursor: pointer;
}

.ui-chip--interactive:hover {
  background: var(--color-bg-tint);
}

.ui-chip--selected {
  background: var(--color-bg-tint-strong);
  border-color: var(--color-primary);
  color: var(--color-primary-dark);
  font-weight: 600;
}

.ui-chip:disabled {
  color: var(--color-text-disabled);
  cursor: default;
}

.ui-chip__image {
  width: 28px;
  height: 28px;
  border-radius: 50%;
  object-fit: cover;
}

.ui-chip__label {
  overflow: hidden;
  text-overflow: ellipsis;
}
</style>
