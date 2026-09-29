<script setup lang="ts">
import { computed, ref } from 'vue'
import type { ScanResult } from '@/services/types/api-types'

interface DebugPanelProps {
  result: ScanResult
}

const { result } = defineProps<DebugPanelProps>()

const isExpanded = ref(false)

const format = (value: number | null, digits = 2): string => {
  if (value === null) {
    return '—'
  }
  return value.toFixed(digits)
}

const summary = computed(() => [
  result.match.status,
  `top1 ${format(result.match.top1?.confidence ?? null)}`,
  `Δ ${format(result.match.margin)}`,
  `F1 ${format(result.match.f1Top1)}/${format(result.match.f1Top5)}`,
  `${format(result.match.latencyMs, 0)} ms`
].join(' · '))

const toggle = (): void => {
  isExpanded.value = !isExpanded.value
}
</script>

<template>
  <aside class="debug-panel" aria-label="Метрики распознавания">
    <button class="debug-panel__summary" type="button" :aria-expanded="isExpanded" @click="toggle">
      <span class="debug-panel__tag">debug</span>
      <span class="debug-panel__text">{{ summary }}</span>
      <span class="debug-panel__chevron" aria-hidden="true">{{ isExpanded ? '▴' : '▾' }}</span>
    </button>
    <div v-if="isExpanded" class="debug-panel__details">
      <p class="debug-panel__row"><span>scanId</span><span>{{ result.scanId }}</span></p>
      <ol class="debug-panel__top5">
        <li v-for="candidate in result.match.top5" :key="candidate.slug" class="debug-panel__row">
          <span>{{ candidate.slug }}</span>
          <span>{{ format(candidate.confidence) }}</span>
        </li>
      </ol>
    </div>
  </aside>
</template>

<style scoped>
.debug-panel {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  z-index: 45;
  padding-top: var(--safe-top);
  background: rgb(44 42 40 / 94%);
  color: var(--color-white);
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 11px;
  line-height: 1.4;
}

.debug-panel__summary {
  display: flex;
  align-items: center;
  gap: var(--space-2);
  width: 100%;
  max-width: var(--container-max);
  min-height: 28px;
  margin: 0 auto;
  padding: 0 var(--space-3);
  border: 0;
  background: transparent;
  color: inherit;
  font: inherit;
  text-align: left;
  cursor: pointer;
}

.debug-panel__tag {
  flex-shrink: 0;
  padding: 0 var(--space-1);
  border-radius: 4px;
  background: var(--color-primary);
  font-weight: 700;
}

.debug-panel__text {
  flex: 1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.debug-panel__chevron {
  flex-shrink: 0;
}

.debug-panel__details {
  max-width: var(--container-max);
  margin: 0 auto;
  padding: 0 var(--space-3) var(--space-2);
  border-top: 1px solid rgb(255 255 255 / 16%);
}

.debug-panel__row {
  display: flex;
  justify-content: space-between;
  gap: var(--space-2);
}

.debug-panel__row span:first-child {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.debug-panel__top5 {
  margin: 0;
  padding: 0;
  list-style: none;
}
</style>
