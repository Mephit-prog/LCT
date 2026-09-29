<script setup lang="ts">
import { computed, ref } from 'vue'
import type { Wine } from '@/services/types/api-types'

interface WineAttributesProps {
  wine: Wine
}

interface AttributeCell {
  key: string
  label: string
  value: string
  icon: string | null
  isExpandable: boolean
}

const { wine } = defineProps<WineAttributesProps>()

const isBlendExpanded = ref(false)
const blendThreshold = 3

const grapesValue = computed((): string | null => {
  const names = wine.grapes.map((grape) => grape.name).filter((name) => name.trim() !== '')
  if (names.length === 0) {
    return null
  }
  if (names.length <= blendThreshold || isBlendExpanded.value) {
    return names.join(', ')
  }
  return `Бленд: ${names.slice(0, 2).join(', ')} и ещё ${names.length - 2}`
})

const isBlend = computed(() => wine.grapes.length > blendThreshold)

const categoryValue = computed((): string | null => {
  const category = wine.category.name.trim()
  const color = wine.color?.trim() ?? ''
  if (category === '' && color === '') {
    return null
  }
  if (category === '') {
    return color
  }
  if (color === '' || category.toLowerCase().includes(color.toLowerCase())) {
    return category
  }
  return `${category}, ${color}`
})

const cells = computed((): AttributeCell[] => {
  const list: Array<AttributeCell | null> = [
    wine.region.name.trim() === ''
      ? null
      : { key: 'region', label: 'Регион', value: wine.region.name, icon: wine.region.image?.url ?? null, isExpandable: false },
    grapesValue.value === null
      ? null
      : { key: 'grapes', label: 'Сорт винограда', value: grapesValue.value, icon: null, isExpandable: isBlend.value },
    categoryValue.value === null
      ? null
      : { key: 'category', label: 'Категория и цвет', value: categoryValue.value, icon: null, isExpandable: false },
    wine.temperature === null || wine.temperature.trim() === ''
      ? null
      : { key: 'temperature', label: 'Температура подачи', value: `${wine.temperature} °C`, icon: null, isExpandable: false },
    wine.alcohol === null
      ? null
      : { key: 'alcohol', label: 'Крепость', value: `${wine.alcohol} %`, icon: null, isExpandable: false }
  ]
  return list.filter((cell): cell is AttributeCell => cell !== null)
})

const toggleBlend = (): void => {
  isBlendExpanded.value = !isBlendExpanded.value
}
</script>

<template>
  <section v-if="cells.length > 0" class="wine-attributes" aria-label="Ключевые атрибуты">
    <dl class="wine-attributes__grid">
      <div v-for="cell in cells" :key="cell.key" class="wine-attributes__cell" :data-attribute="cell.key">
        <dt class="wine-attributes__label">{{ cell.label }}</dt>
        <dd class="wine-attributes__value">
          <img v-if="cell.icon !== null" class="wine-attributes__icon" :src="cell.icon" alt="" width="20" height="20" loading="lazy" decoding="async">
          <span>{{ cell.value }}</span>
          <button v-if="cell.isExpandable" class="wine-attributes__expand" type="button" @click="toggleBlend">
            {{ isBlendExpanded ? 'Свернуть' : 'Все сорта' }}
          </button>
        </dd>
      </div>
    </dl>
  </section>
</template>

<style scoped>
.wine-attributes {
  padding: var(--space-4);
}

.wine-attributes__grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: var(--space-3);
  margin: 0;
}

.wine-attributes__cell {
  padding: var(--space-3);
  border-radius: var(--radius-md);
  background: var(--color-bg-soft);
}

.wine-attributes__label {
  margin-bottom: var(--space-1);
  color: var(--color-text-muted);
  font-size: var(--text-xs);
}

.wine-attributes__value {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: var(--space-1) var(--space-2);
  margin: 0;
  font-size: var(--text-sm);
  font-weight: 600;
  overflow-wrap: anywhere;
}

.wine-attributes__icon {
  width: 20px;
  height: 20px;
  object-fit: contain;
}

.wine-attributes__expand {
  padding: 0;
  border: 0;
  background: transparent;
  color: var(--color-primary);
  font-size: var(--text-xs);
  font-weight: 600;
  cursor: pointer;
  min-height: 24px;
}
</style>
