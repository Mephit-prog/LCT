<script setup lang="ts">
interface UiSkeletonProps {
  width?: string
  height?: string
  radius?: string
  isCircle?: boolean
}

const { width = '100%', height = '16px', radius = 'var(--radius-sm)', isCircle = false } = defineProps<UiSkeletonProps>()
</script>

<template>
  <span
    class="ui-skeleton"
    :style="{ width, height, borderRadius: isCircle ? '50%' : radius }"
    aria-hidden="true"
  />
</template>

<style scoped>
.ui-skeleton {
  display: block;
  background: linear-gradient(90deg, var(--color-border-soft) 0%, var(--color-bg-soft) 50%, var(--color-border-soft) 100%);
  background-size: 200% 100%;
  animation: ui-skeleton-shimmer 1.4s ease-in-out infinite;
}

@keyframes ui-skeleton-shimmer {
  from {
    background-position: 200% 0;
  }

  to {
    background-position: -200% 0;
  }
}
</style>
