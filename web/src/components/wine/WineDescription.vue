<script setup lang="ts">
import { computed, ref } from 'vue'

interface WineDescriptionProps {
  text: string | null
}

const { text } = defineProps<WineDescriptionProps>()

const isExpanded = ref(false)
const collapsedThreshold = 220

const hasText = computed(() => text !== null && text.trim() !== '')
const isLong = computed(() => hasText.value && (text?.length ?? 0) > collapsedThreshold)

const toggle = (): void => {
  isExpanded.value = !isExpanded.value
}
</script>

<template>
  <section v-if="hasText" class="wine-description" aria-labelledby="description-title">
    <h2 id="description-title" class="section-title">Описание</h2>
    <p class="wine-description__text" :class="{ 'wine-description__text--clamped': isLong && !isExpanded }">{{ text }}</p>
    <button v-if="isLong" class="wine-description__toggle" type="button" :aria-expanded="isExpanded" @click="toggle">
      {{ isExpanded ? 'Свернуть' : 'Читать полностью' }}
    </button>
  </section>
</template>

<style scoped>
.wine-description {
  padding: var(--space-4);
}

.wine-description__text {
  color: var(--color-text);
}

.wine-description__text--clamped {
  display: -webkit-box;
  -webkit-box-orient: vertical;
  -webkit-line-clamp: 4;
  line-clamp: 4;
  overflow: hidden;
}

.wine-description__toggle {
  min-height: var(--tap-size);
  padding: 0;
  border: 0;
  background: transparent;
  color: var(--color-primary);
  font-weight: 600;
  cursor: pointer;
}
</style>
