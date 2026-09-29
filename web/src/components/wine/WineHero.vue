<script setup lang="ts">
import { computed } from 'vue'
import type { Wine } from '@/services/types/api-types'
import { env } from '@/services/constants/env'

interface WineHeroProps {
  wine: Wine
}

const { wine } = defineProps<WineHeroProps>()

const background = computed(() => wine.category.backgroundGradient ?? 'var(--color-bg-soft)')
const manufacturerHref = computed(() => `${env.portalBaseUrl}/manufacturers/${wine.manufacturer.slug}`)
const ratingLabel = computed(() => wine.publicRating === null ? null : wine.publicRating.toFixed(2))
</script>

<template>
  <section class="wine-hero" :style="{ background }">
    <div class="wine-hero__image-wrap">
      <img
        class="wine-hero__image"
        :src="wine.image.url"
        :alt="wine.image.altText ?? wine.title"
        width="180"
        height="320"
        decoding="async"
        fetchpriority="high"
      >
    </div>
    <h1 class="wine-hero__title">{{ wine.title }}</h1>
    <a class="wine-hero__manufacturer" :href="manufacturerHref" rel="noopener">{{ wine.manufacturer.name }}</a>
    <div v-if="ratingLabel !== null || wine.roskachestvoRating !== null" class="wine-hero__ratings">
      <span v-if="ratingLabel !== null" class="wine-hero__rating" aria-label="Оценка пользователей">
        <span class="wine-hero__rating-icon" aria-hidden="true">★</span>
        <span>{{ ratingLabel }}</span>
      </span>
      <span v-if="wine.roskachestvoRating !== null" class="wine-hero__roskachestvo">
        <span class="wine-hero__roskachestvo-label">Роскачество</span>
        <span class="wine-hero__roskachestvo-score">{{ wine.roskachestvoRating.score }} · {{ wine.roskachestvoRating.year }}</span>
      </span>
    </div>
  </section>
</template>

<style scoped>
.wine-hero {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: var(--space-2);
  padding: var(--space-4) var(--space-4) var(--space-6);
  text-align: center;
}

.wine-hero__image-wrap {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 100%;
  height: 280px;
  margin-bottom: var(--space-2);
}

.wine-hero__image {
  width: auto;
  height: 100%;
  object-fit: contain;
  filter: drop-shadow(0 12px 18px rgb(44 42 40 / 18%));
}

.wine-hero__title {
  font-size: var(--text-2xl);
  line-height: 32px;
}

.wine-hero__manufacturer {
  color: var(--color-primary);
  font-size: var(--text-sm);
  font-weight: 600;
  min-height: 24px;
}

.wine-hero__ratings {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: var(--space-2);
  margin-top: var(--space-2);
}

.wine-hero__rating {
  display: inline-flex;
  align-items: center;
  gap: var(--space-1);
  padding: var(--space-1) var(--space-3);
  border-radius: var(--radius-pill);
  background: rgb(255 255 255 / 70%);
  font-weight: 600;
}

.wine-hero__rating-icon {
  color: var(--color-primary-dark);
}

.wine-hero__roskachestvo {
  display: inline-flex;
  align-items: center;
  gap: var(--space-2);
  padding: var(--space-1) var(--space-3);
  border: 1px solid var(--color-primary-dark);
  border-radius: var(--radius-pill);
  color: var(--color-primary-dark);
  font-size: var(--text-sm);
}

.wine-hero__roskachestvo-label {
  font-weight: 600;
}
</style>
