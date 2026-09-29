<script setup lang="ts">
import { computed } from 'vue'
import UiBottomSheet from '@/components/ui/UiBottomSheet.vue'
import UiButton from '@/components/ui/UiButton.vue'
import type { WineSummary } from '@/services/types/api-types'

interface UncertainSheetProps {
  candidates: WineSummary[]
  scanId: string
}

const { candidates, scanId } = defineProps<UncertainSheetProps>()

const isOpen = defineModel<boolean>({ default: false })

const emit = defineEmits<{ confirm: [], select: [slug: string] }>()

const visibleCandidates = computed(() => candidates.slice(0, 3))

const differentiator = (candidate: WineSummary): string => {
  return [candidate.category, candidate.region].filter((part) => part.trim() !== '').join(' · ')
}

const onConfirm = (): void => {
  emit('confirm')
  isOpen.value = false
}

const onSelect = (slug: string): void => {
  emit('select', slug)
  isOpen.value = false
}
</script>

<template>
  <UiBottomSheet v-model="isOpen" title="Проверьте: это ваше вино?" :is-modal="false" max-height="40dvh">
    <ul class="uncertain__list">
      <li v-for="candidate in visibleCandidates" :key="candidate.slug">
        <RouterLink
          class="uncertain__item"
          :to="{ name: 'wine', params: { slug: candidate.slug }, query: { scan: scanId } }"
          @click="onSelect(candidate.slug)"
        >
          <img class="uncertain__image" :src="candidate.image.url" :alt="candidate.image.altText ?? candidate.title" width="27" height="48" loading="lazy" decoding="async">
          <span class="uncertain__body">
            <span class="uncertain__title">{{ candidate.title }}</span>
            <span class="uncertain__meta">{{ candidate.manufacturer }} · {{ differentiator(candidate) }}</span>
          </span>
          <span class="uncertain__arrow" aria-hidden="true">›</span>
        </RouterLink>
      </li>
    </ul>
    <UiButton size="lg" is-block class="uncertain__confirm" @click="onConfirm">Да, это оно</UiButton>
  </UiBottomSheet>
</template>

<style scoped>
.uncertain__list {
  display: flex;
  flex-direction: column;
  gap: var(--space-2);
  margin-bottom: var(--space-3);
}

.uncertain__item {
  display: flex;
  align-items: center;
  gap: var(--space-3);
  min-height: var(--tap-size);
  padding: var(--space-1) var(--space-3);
  border-radius: var(--radius-md);
  background: var(--color-bg-tint);
  color: var(--color-text);
}

.uncertain__image {
  width: 27px;
  height: 48px;
  object-fit: contain;
}

.uncertain__body {
  display: flex;
  flex-direction: column;
  min-width: 0;
  flex: 1;
}

.uncertain__title {
  font-weight: 600;
  font-size: var(--text-sm);
}

.uncertain__meta {
  color: var(--color-text-muted);
  font-size: var(--text-xs);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.uncertain__arrow {
  color: var(--color-text-muted);
  font-size: var(--text-lg);
}
</style>
