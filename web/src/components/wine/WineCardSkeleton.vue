<script setup lang="ts">
import UiSkeleton from '@/components/ui/UiSkeleton.vue'
</script>

<template>
  <div class="wine-skeleton" aria-busy="true" aria-label="Загрузка карточки">
    <div class="wine-skeleton__hero">
      <UiSkeleton width="100px" height="240px" radius="var(--radius-md)" />
      <UiSkeleton width="70%" height="28px" />
      <UiSkeleton width="40%" height="16px" />
    </div>
    <div class="wine-skeleton__grid">
      <UiSkeleton v-for="index in 4" :key="index" height="64px" radius="var(--radius-md)" />
    </div>
    <div class="wine-skeleton__lines">
      <UiSkeleton height="14px" />
      <UiSkeleton height="14px" />
      <UiSkeleton width="60%" height="14px" />
    </div>
  </div>
</template>

<style scoped>
.wine-skeleton {
  display: flex;
  flex-direction: column;
  gap: var(--space-4);
}

.wine-skeleton__hero {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: var(--space-3);
  padding: var(--space-6) var(--space-4);
  background: var(--color-bg-soft);
}

.wine-skeleton__grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: var(--space-3);
  padding: 0 var(--space-4);
}

.wine-skeleton__lines {
  display: flex;
  flex-direction: column;
  gap: var(--space-2);
  padding: 0 var(--space-4);
}
</style>
