<script setup lang="ts">
import { computed } from 'vue'
import UiChip from '@/components/ui/UiChip.vue'
import { localizeReasons } from '@/services/constants/reasons'

interface ReasonChipsProps {
  reasons: readonly string[]
  label?: string
}

const { reasons, label = 'Почему предложено' } = defineProps<ReasonChipsProps>()

const labels = computed(() => localizeReasons(reasons))
</script>

<template>
  <div v-if="labels.length > 0" class="reason-chips">
    <span class="reason-chips__label">{{ label }}:</span>
    <ul class="reason-chips__list">
      <li v-for="reason in labels" :key="reason">
        <UiChip :label="reason" size="sm" />
      </li>
    </ul>
  </div>
</template>

<style scoped>
.reason-chips {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: var(--space-1) var(--space-2);
}

.reason-chips__label {
  color: var(--color-text-muted);
  font-size: var(--text-xs);
}

.reason-chips__list {
  display: flex;
  flex-wrap: wrap;
  gap: var(--space-1);
}
</style>
