<script setup lang="ts">
import UiChip from '@/components/ui/UiChip.vue'
import type { Wine } from '@/services/types/api-types'

interface DishesRowProps {
  dishes: Wine['dishes']
}

const { dishes } = defineProps<DishesRowProps>()
</script>

<template>
  <section v-if="dishes.length > 0" class="dishes-row" aria-labelledby="dishes-title">
    <h2 id="dishes-title" class="section-title dishes-row__title">Сочетание с блюдами</h2>
    <ul class="h-scroll dishes-row__list">
      <li v-for="dish in dishes" :key="dish.name" class="dishes-row__item">
        <UiChip :label="dish.name" :image-url="dish.image?.url ?? null" />
      </li>
    </ul>
  </section>
</template>

<style scoped>
.dishes-row {
  padding: var(--space-4) 0;
}

.dishes-row__title {
  padding: 0 var(--space-4);
}

.dishes-row__list {
  padding-left: var(--space-4);
  padding-right: var(--space-4);
}

.dishes-row__item {
  flex-shrink: 0;
  scroll-snap-align: start;
}
</style>
