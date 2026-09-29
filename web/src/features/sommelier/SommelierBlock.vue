<script setup lang="ts">
import { computed, ref } from 'vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiChip from '@/components/ui/UiChip.vue'
import { track } from '@/services/analytics/events'
import { isAbortError } from '@/services/apis/errors'
import type { SommelierMode, Wine, WineSummary } from '@/services/types/api-types'
import { buildFallbackReply, isDishCode, isTasteCode, offerDishAlternative, offerTasteAlternative } from './sommelier-fallback'
import { getFollowUp, labelOf, occasionOptions, type QuickOption } from './sommelier-options'
import { findDishAlternatives, findTasteAlternatives } from './sommelier-suggestions'
import { useSommelierStream } from './useSommelierStream'

interface SommelierBlockProps {
  mode: SommelierMode
  wine?: Wine | null
  scanId?: string | null
  candidates?: WineSummary[]
}

const { mode, wine = null, scanId = null, candidates = [] } = defineProps<SommelierBlockProps>()

type Step = 'occasion' | 'detail' | 'answer'

const step = ref<Step>('occasion')
const occasion = ref<string | null>(null)
const detail = ref<string | null>(null)
const { displayedText, recommendations, temperature, status, isTyping, ask, reset } = useSommelierStream()
let searchController: AbortController | null = null

const subtitle = computed(() => mode === 'pairing'
  ? 'Подскажем, брать ли это вино и к чему его подать'
  : 'Расскажите, к чему подбираете вино — предложим варианты')

const followUp = computed(() => occasion.value === null ? null : getFollowUp(occasion.value))
const occasionLabel = computed(() => labelOf(occasionOptions, occasion.value))
const detailLabel = computed(() => followUp.value === null ? null : labelOf(followUp.value.options, detail.value))
const isAnswerReady = computed(() => status.value === 'done' || status.value === 'fallback')
const answerContext = computed(() => [occasionLabel.value, detailLabel.value].filter((label) => label !== null).join(' · '))

const recommendationRoute = (slug: string): { name: string, params: { slug: string }, query: Record<string, string> } => ({
  name: 'wine',
  params: { slug },
  query: scanId === null ? {} : { scan: scanId }
})

const startAnswer = async (): Promise<void> => {
  if (occasion.value === null) {
    return
  }
  step.value = 'answer'
  const answers = { occasion: occasion.value, dish: detail.value }
  track({ name: 'sommelier_answered', mode, occasion: answers.occasion, dish: answers.dish, slug: wine?.slug ?? null })
  let fallback = buildFallbackReply({ mode, wine, candidates }, answers)
  const current = wine
  const dish = answers.dish
  if (current !== null && dish !== null && fallback.recommendations.length === 0 && fallback.text.startsWith('Лучше взять другое') && (isTasteCode(dish) || isDishCode(dish))) {
    searchController?.abort()
    const controller = new AbortController()
    searchController = controller
    try {
      if (isTasteCode(dish)) {
        const suggestions = await findTasteAlternatives(current, dish, controller.signal)
        if (searchController !== controller) {
          return
        }
        fallback = offerTasteAlternative(fallback, suggestions, dish)
      } else if (isDishCode(dish)) {
        const suggestions = await findDishAlternatives(current, dish, controller.signal)
        if (searchController !== controller) {
          return
        }
        fallback = offerDishAlternative(fallback, suggestions, dish)
      }
    } catch (error) {
      if (isAbortError(error) || searchController !== controller) {
        return
      }
    }
  }
  await ask({ mode, scanId, slug: wine?.slug ?? null, answers }, fallback)
}

const onOccasion = (option: QuickOption): void => {
  occasion.value = option.code
  detail.value = null
  track({ name: 'sommelier_started', mode, slug: wine?.slug ?? null })
  if (getFollowUp(option.code) === null) {
    void startAnswer()
    return
  }
  step.value = 'detail'
}

const onDetail = (option: QuickOption): void => {
  detail.value = option.code
  void startAnswer()
}

const onRestart = (): void => {
  searchController?.abort()
  searchController = null
  reset()
  occasion.value = null
  detail.value = null
  step.value = 'occasion'
}

const onRecommendationOpen = (slug: string): void => {
  track({ name: 'sommelier_recommendation_opened', mode, slug })
}
</script>

<template>
  <section class="sommelier" aria-labelledby="sommelier-title">
    <header class="sommelier__header">
      <span class="sommelier__badge" aria-hidden="true">✦</span>
      <div>
        <h2 id="sommelier-title" class="sommelier__title">Цифровой сомелье</h2>
        <p class="sommelier__subtitle">{{ subtitle }}</p>
      </div>
    </header>

    <div v-if="step === 'occasion'" class="sommelier__question">
      <p class="sommelier__prompt">К чему подбираете вино?</p>
      <ul class="sommelier__chips">
        <li v-for="option in occasionOptions" :key="option.code">
          <UiChip :label="option.label" is-interactive @click="onOccasion(option)" />
        </li>
      </ul>
    </div>

    <div v-else-if="step === 'detail' && followUp !== null" class="sommelier__question">
      <p class="sommelier__context">{{ occasionLabel }}</p>
      <p class="sommelier__prompt">{{ followUp.question }}</p>
      <ul class="sommelier__chips">
        <li v-for="option in followUp.options" :key="option.code">
          <UiChip :label="option.label" is-interactive @click="onDetail(option)" />
        </li>
      </ul>
      <UiButton variant="transparent" @click="onRestart">Назад</UiButton>
    </div>

    <div v-else class="sommelier__answer" aria-live="polite">
      <p class="sommelier__context">{{ answerContext }}</p>
      <div class="sommelier__message">
        <p class="sommelier__text">
          <span>{{ displayedText }}</span>
          <span v-if="isTyping" class="sommelier__caret" aria-hidden="true" />
        </p>
        <p v-if="temperature !== null && isAnswerReady" class="sommelier__temperature">
          Температура подачи: {{ temperature }} °C
        </p>
        <p v-if="status === 'fallback'" class="sommelier__note">Ответ подготовлен на основе данных карточки</p>
      </div>
      <ul v-if="recommendations.length > 0" class="sommelier__recommendations">
        <li v-for="item in recommendations" :key="item.wine.slug">
          <RouterLink class="sommelier__recommendation" :to="recommendationRoute(item.wine.slug)" @click="onRecommendationOpen(item.wine.slug)">
            <img class="sommelier__recommendation-image" :src="item.wine.image.url" :alt="item.wine.image.altText ?? item.wine.title" loading="lazy" decoding="async" width="41" height="72">
            <span class="sommelier__recommendation-body">
              <span class="sommelier__recommendation-title">{{ item.wine.title }}</span>
              <span class="sommelier__recommendation-manufacturer">{{ item.wine.manufacturer }}</span>
              <UiChip :label="item.reason" size="sm" is-selected />
            </span>
          </RouterLink>
        </li>
      </ul>
      <UiButton v-if="isAnswerReady" variant="tertiary" @click="onRestart">Задать другой вопрос</UiButton>
    </div>
  </section>
</template>

<style scoped>
.sommelier {
  margin: var(--space-4);
  padding: var(--space-4);
  border-radius: var(--radius-lg);
  background: var(--color-bg-soft);
}

.sommelier__header {
  display: flex;
  gap: var(--space-3);
  align-items: flex-start;
  margin-bottom: var(--space-4);
}

.sommelier__badge {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
  width: 36px;
  height: 36px;
  border-radius: 50%;
  background: var(--color-primary);
  color: var(--color-white);
}

.sommelier__title {
  font-size: var(--text-lg);
}

.sommelier__subtitle {
  color: var(--color-text-muted);
  font-size: var(--text-sm);
}

.sommelier__question,
.sommelier__answer {
  display: flex;
  flex-direction: column;
  gap: var(--space-3);
}

.sommelier__prompt {
  font-weight: 600;
}

.sommelier__context {
  color: var(--color-text-muted);
  font-size: var(--text-xs);
  text-transform: uppercase;
  letter-spacing: 0.06em;
}

.sommelier__chips {
  display: flex;
  flex-wrap: wrap;
  gap: var(--space-2);
}

.sommelier__message {
  display: flex;
  flex-direction: column;
  gap: var(--space-2);
  padding: var(--space-3) var(--space-4);
  border-radius: var(--radius-md) var(--radius-md) var(--radius-md) var(--radius-sm);
  background: var(--color-white);
  box-shadow: var(--shadow-card);
  min-height: 56px;
}

.sommelier__text {
  white-space: pre-wrap;
}

.sommelier__caret {
  display: inline-block;
  width: 2px;
  height: 1em;
  margin-left: 2px;
  vertical-align: text-bottom;
  background: var(--color-primary);
  animation: sommelier-caret 0.9s steps(2) infinite;
}

.sommelier__temperature {
  color: var(--color-primary-dark);
  font-size: var(--text-sm);
  font-weight: 600;
}

.sommelier__note {
  color: var(--color-text-muted);
  font-size: var(--text-xs);
}

.sommelier__recommendations {
  display: flex;
  flex-direction: column;
  gap: var(--space-2);
}

.sommelier__recommendation {
  display: flex;
  gap: var(--space-3);
  align-items: center;
  min-height: var(--tap-size);
  padding: var(--space-2) var(--space-3);
  border-radius: var(--radius-md);
  background: var(--color-white);
  color: var(--color-text);
}

.sommelier__recommendation-image {
  width: 41px;
  height: 72px;
  object-fit: contain;
}

.sommelier__recommendation-body {
  display: flex;
  flex-direction: column;
  gap: var(--space-1);
  min-width: 0;
  align-items: flex-start;
}

.sommelier__recommendation-title {
  font-weight: 600;
  font-size: var(--text-sm);
}

.sommelier__recommendation-manufacturer {
  color: var(--color-text-muted);
  font-size: var(--text-xs);
}

@keyframes sommelier-caret {
  from {
    opacity: 1;
  }

  to {
    opacity: 0;
  }
}
</style>
