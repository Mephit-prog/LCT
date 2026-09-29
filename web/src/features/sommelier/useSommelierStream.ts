import { computed, onScopeDispose, readonly, ref, type ComputedRef, type Ref } from 'vue'
import { apis } from '@/services/apis'
import { isAbortError } from '@/services/apis/errors'
import { logError, logWarn } from '@/services/analytics/logger'
import { env } from '@/services/constants/env'
import { typingIntervalMs } from '@/services/constants/timings'
import type { SommelierRecommendation, SommelierRequest } from '@/services/types/api-types'
import type { SommelierReply } from './sommelier-fallback'

export type SommelierStreamStatus = 'idle' | 'streaming' | 'done' | 'fallback'

export interface UseSommelierStream {
  displayedText: Readonly<Ref<string>>
  recommendations: Readonly<Ref<readonly SommelierRecommendation[]>>
  temperature: Readonly<Ref<string | null>>
  status: Readonly<Ref<SommelierStreamStatus>>
  isTyping: ComputedRef<boolean>
  ask: (request: SommelierRequest, fallback: SommelierReply) => Promise<void>
  reset: () => void
}

const charsPerTick = 3

const prefersReducedMotion = (): boolean => {
  if (typeof matchMedia !== 'function') {
    return false
  }
  return matchMedia('(prefers-reduced-motion: reduce)').matches
}

export const useSommelierStream = (): UseSommelierStream => {
  const displayedText = ref('')
  const recommendations = ref<SommelierRecommendation[]>([])
  const temperature = ref<string | null>(null)
  const status = ref<SommelierStreamStatus>('idle')

  let fullText = ''
  let isSourceFinished = false
  let controller: AbortController | null = null
  let typingTimer: ReturnType<typeof setInterval> | null = null
  let timeoutTimer: ReturnType<typeof setTimeout> | null = null
  let isInstant = false

  const isTyping = computed(() => displayedText.value.length < fullText.length || (status.value === 'streaming' && !isSourceFinished))

  const stopTyping = (): void => {
    if (typingTimer !== null) {
      clearInterval(typingTimer)
      typingTimer = null
    }
  }

  const clearTimeoutTimer = (): void => {
    if (timeoutTimer !== null) {
      clearTimeout(timeoutTimer)
      timeoutTimer = null
    }
  }

  const finishIfCaughtUp = (): void => {
    if (!isSourceFinished || displayedText.value.length < fullText.length) {
      return
    }
    stopTyping()
    if (status.value === 'streaming') {
      status.value = 'done'
    }
  }

  const tickTyping = (): void => {
    if (displayedText.value.length < fullText.length) {
      displayedText.value = fullText.slice(0, displayedText.value.length + charsPerTick)
    }
    finishIfCaughtUp()
  }

  const ensureTyping = (): void => {
    if (isInstant) {
      displayedText.value = fullText
      finishIfCaughtUp()
      return
    }
    if (typingTimer === null) {
      typingTimer = setInterval(tickTyping, typingIntervalMs)
    }
  }

  const applyFallback = (fallback: SommelierReply): void => {
    fullText = fallback.text
    displayedText.value = isInstant ? fullText : displayedText.value.slice(0, Math.min(displayedText.value.length, fullText.length))
    recommendations.value = fallback.recommendations
    temperature.value = fallback.temperature
    status.value = 'fallback'
    isSourceFinished = true
    ensureTyping()
  }

  const reset = (): void => {
    controller?.abort()
    controller = null
    clearTimeoutTimer()
    stopTyping()
    fullText = ''
    isSourceFinished = false
    displayedText.value = ''
    recommendations.value = []
    temperature.value = null
    status.value = 'idle'
  }

  const ask = async (request: SommelierRequest, fallback: SommelierReply): Promise<void> => {
    reset()
    isInstant = prefersReducedMotion()
    status.value = 'streaming'
    temperature.value = fallback.temperature
    const ownController = new AbortController()
    controller = ownController

    let didTimeout = false
    timeoutTimer = setTimeout(() => {
      didTimeout = true
      ownController.abort()
    }, env.sommelierTimeoutMs)

    try {
      for await (const event of apis.sommelier(request, ownController.signal)) {
        if (ownController.signal.aborted) {
          break
        }
        if (event.type === 'token') {
          fullText += event.text
          ensureTyping()
        } else if (event.type === 'recommendation') {
          recommendations.value = [...recommendations.value, event.recommendation]
        } else {
          isSourceFinished = true
        }
      }
      if (controller !== ownController) {
        return
      }
      clearTimeoutTimer()
      if (didTimeout || fullText.trim() === '') {
        logWarn('sommelier', didTimeout ? 'Таймаут потока, показан fallback' : 'Пустой ответ, показан fallback')
        applyFallback(fallback)
        return
      }
      isSourceFinished = true
      ensureTyping()
      finishIfCaughtUp()
    } catch (error) {
      if (controller !== ownController) {
        return
      }
      clearTimeoutTimer()
      if (!isAbortError(error) || didTimeout) {
        logError('sommelier', error, { didTimeout })
      }
      applyFallback(fallback)
    }
  }

  onScopeDispose(() => {
    controller?.abort()
    clearTimeoutTimer()
    stopTyping()
  })

  return {
    displayedText: readonly(displayedText),
    recommendations: readonly(recommendations),
    temperature: readonly(temperature),
    status: readonly(status),
    isTyping,
    ask,
    reset
  }
}
