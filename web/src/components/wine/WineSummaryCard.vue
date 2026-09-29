<script setup lang="ts">
import { computed } from 'vue'
import type { RouteLocationRaw } from 'vue-router'
import type { WineSummary } from '@/services/types/api-types'

interface WineSummaryCardProps {
  wine: WineSummary
  to: RouteLocationRaw
  variant?: 'carousel' | 'list'
}

const { wine, to, variant = 'carousel' } = defineProps<WineSummaryCardProps>()

const emit = defineEmits<{ open: [slug: string] }>()

const rating = computed(() => wine.publicRating === null ? null : wine.publicRating.toFixed(2))
const meta = computed(() => [wine.region, wine.category].filter((part) => part.trim() !== '').join(' · '))

const onClick = (): void => emit('open', wine.slug)
</script>

<template>
  <RouterLink class="summary-card" :class="`summary-card--${variant}`" :to="to" @click="onClick">
    <span class="summary-card__image-wrap">
      <img class="summary-card__image" :src="wine.image.url" :alt="wine.image.altText ?? wine.title" loading="lazy" decoding="async" width="90" height="160">
    </span>
    <span class="summary-card__body">
      <span class="summary-card__title">{{ wine.title }}</span>
      <span class="summary-card__manufacturer">{{ wine.manufacturer }}</span>
      <span v-if="variant === 'list' && meta !== ''" class="summary-card__meta">{{ meta }}</span>
      <span v-if="rating !== null" class="summary-card__rating">
        <span aria-hidden="true">★</span>
        <span>{{ rating }}</span>
      </span>
      <slot />
    </span>
  </RouterLink>
</template>

<style scoped>
.summary-card {
  display: flex;
  color: var(--color-text);
  border-radius: var(--radius-md);
  background: var(--color-white);
  border: 1px solid var(--color-border-soft);
  -webkit-tap-highlight-color: transparent;
  transition: border-color var(--duration-fast) var(--ease-out);
}

.summary-card:hover {
  border-color: var(--color-primary);
}

.summary-card--carousel {
  flex-direction: column;
  flex: 0 0 148px;
  padding: var(--space-3);
  scroll-snap-align: start;
}

.summary-card--list {
  align-items: stretch;
  gap: var(--space-3);
  padding: var(--space-3);
}

.summary-card__image-wrap {
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
  border-radius: var(--radius-sm);
  background: var(--color-bg-soft);
}

.summary-card--carousel .summary-card__image-wrap {
  height: 140px;
  margin-bottom: var(--space-2);
}

.summary-card--list .summary-card__image-wrap {
  width: 72px;
  min-height: 120px;
}

.summary-card__image {
  width: auto;
  height: 120px;
  object-fit: contain;
}

.summary-card--list .summary-card__image {
  height: 104px;
}

.summary-card__body {
  display: flex;
  flex-direction: column;
  gap: var(--space-1);
  min-width: 0;
  flex: 1;
}

.summary-card__title {
  font-family: var(--font-heading);
  font-weight: 600;
  font-size: var(--text-sm);
  line-height: 1.3;
  display: -webkit-box;
  -webkit-box-orient: vertical;
  -webkit-line-clamp: 2;
  line-clamp: 2;
  overflow: hidden;
}

.summary-card--list .summary-card__title {
  font-size: var(--text-md);
}

.summary-card__manufacturer,
.summary-card__meta {
  color: var(--color-text-muted);
  font-size: var(--text-xs);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.summary-card__rating {
  display: inline-flex;
  gap: var(--space-1);
  color: var(--color-primary-dark);
  font-size: var(--text-xs);
  font-weight: 600;
}
</style>
