<script setup lang="ts">
interface PageHeaderProps {
  backLabel: string
  backTo: string
}

const { backLabel, backTo } = defineProps<PageHeaderProps>()
</script>

<template>
  <nav class="page-header" aria-label="Навигация">
    <RouterLink class="page-header__back" :to="backTo">
      <span aria-hidden="true">←</span>
      <span>{{ backLabel }}</span>
    </RouterLink>
    <slot />
  </nav>
</template>

<style scoped>
.page-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: var(--space-2);
  min-height: var(--tap-size);
  padding: var(--space-2) var(--space-4);
  padding-top: calc(var(--space-2) + var(--safe-top));
}

.page-header__back {
  display: inline-flex;
  align-items: center;
  gap: var(--space-2);
  min-height: var(--tap-size);
  color: var(--color-text-muted);
  font-size: var(--text-sm);
}

.page-header__back:hover {
  color: var(--color-primary);
}
</style>
