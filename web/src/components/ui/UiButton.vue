<script setup lang="ts">
export type UiButtonVariant = 'primary' | 'secondary' | 'tertiary' | 'transparent'
export type UiButtonSize = 'md' | 'lg'

interface UiButtonProps {
  variant?: UiButtonVariant
  size?: UiButtonSize
  type?: 'button' | 'submit'
  isDisabled?: boolean
  isLoading?: boolean
  isBlock?: boolean
  ariaLabel?: string
}

const {
  variant = 'primary',
  size = 'md',
  type = 'button',
  isDisabled = false,
  isLoading = false,
  isBlock = false,
  ariaLabel
} = defineProps<UiButtonProps>()

const emit = defineEmits<{ click: [event: MouseEvent] }>()

const onClick = (event: MouseEvent): void => {
  if (isDisabled || isLoading) {
    event.preventDefault()
    return
  }
  emit('click', event)
}
</script>

<template>
  <button
    class="ui-button"
    :class="[`ui-button--${variant}`, `ui-button--${size}`, { 'ui-button--block': isBlock, 'ui-button--loading': isLoading }]"
    :type="type"
    :disabled="isDisabled || isLoading"
    :aria-busy="isLoading || undefined"
    :aria-label="ariaLabel"
    @click="onClick"
  >
    <span v-if="isLoading" class="ui-button__loader" aria-hidden="true" />
    <span class="ui-button__content"><slot /></span>
  </button>
</template>

<style scoped>
.ui-button {
  position: relative;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: var(--space-2);
  min-width: var(--tap-size);
  padding: 0 var(--space-4);
  border: 1px solid transparent;
  border-radius: var(--radius-md);
  font-weight: 600;
  font-size: var(--text-md);
  line-height: 1;
  cursor: pointer;
  user-select: none;
  -webkit-tap-highlight-color: transparent;
  transition: background-color var(--duration-fast) var(--ease-out), border-color var(--duration-fast) var(--ease-out), color var(--duration-fast) var(--ease-out);
}

.ui-button--md {
  height: var(--control-height-md);
}

.ui-button--lg {
  height: var(--control-height-lg);
  padding: 0 var(--space-6);
}

.ui-button--block {
  display: flex;
  width: 100%;
}

.ui-button--primary {
  background: var(--color-primary);
  color: var(--color-white);
}

.ui-button--primary:hover,
.ui-button--primary:active {
  background: var(--color-primary-hover);
}

.ui-button--secondary {
  background: var(--color-white);
  border-color: var(--color-primary);
  color: var(--color-primary);
}

.ui-button--secondary:hover,
.ui-button--secondary:active {
  background: var(--color-bg-tint);
}

.ui-button--tertiary {
  background: var(--color-bg-tint);
  color: var(--color-primary);
}

.ui-button--tertiary:hover,
.ui-button--tertiary:active {
  background: var(--color-bg-tint-strong);
}

.ui-button--transparent {
  background: transparent;
  color: var(--color-primary);
  padding: 0 var(--space-2);
}

.ui-button--transparent:hover,
.ui-button--transparent:active {
  background: var(--color-bg-tint);
}

.ui-button:disabled {
  cursor: default;
  background: var(--color-border-soft);
  border-color: transparent;
  color: var(--color-text-disabled);
}

.ui-button--loading .ui-button__content {
  opacity: 0.4;
}

.ui-button__loader {
  position: absolute;
  left: 50%;
  top: 50%;
  width: 40%;
  height: 3px;
  transform: translate(-50%, -50%);
  border-radius: var(--radius-pill);
  background: linear-gradient(90deg, transparent, currentColor, transparent);
  background-size: 200% 100%;
  animation: ui-button-loading 1s linear infinite;
}

@keyframes ui-button-loading {
  from {
    background-position: 200% 0;
  }

  to {
    background-position: -200% 0;
  }
}
</style>
