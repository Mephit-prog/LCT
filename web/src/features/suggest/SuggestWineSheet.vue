<script setup lang="ts">
import { ref } from 'vue'
import UiBottomSheet from '@/components/ui/UiBottomSheet.vue'
import UiButton from '@/components/ui/UiButton.vue'
import { useToast } from '@/composables/useToast'
import { apis } from '@/services/apis'
import { track } from '@/services/analytics/events'
import { logError } from '@/services/analytics/logger'
import { uiMessages } from '@/services/constants/messages'

interface SuggestWineSheetProps {
  scanId: string
  previewUrl: string | null
}

const { scanId, previewUrl } = defineProps<SuggestWineSheetProps>()

const isOpen = defineModel<boolean>({ default: false })

const toast = useToast()
const comment = ref('')
const isSubmitting = ref(false)
const isSubmitted = ref(false)

const onInput = (event: Event): void => {
  const target = event.target
  if (target instanceof HTMLInputElement) {
    comment.value = target.value
  }
}

const onSubmit = async (): Promise<void> => {
  if (isSubmitting.value) {
    return
  }
  isSubmitting.value = true
  try {
    const trimmed = comment.value.trim()
    await apis.suggest(scanId, { comment: trimmed === '' ? null : trimmed })
    track({ name: 'suggest_submitted', scanId })
    isSubmitted.value = true
  } catch (error) {
    logError('suggest', error, { scanId })
    toast.error(uiMessages.suggestFailed)
  } finally {
    isSubmitting.value = false
  }
}

const onClose = (): void => {
  isOpen.value = false
}
</script>

<template>
  <UiBottomSheet v-model="isOpen" title="Предложить вино в каталог">
    <div v-if="isSubmitted" class="suggest__done">
      <p class="suggest__done-title">Спасибо, передадим редакции каталога</p>
      <p class="suggest__hint">Фото уже приложено к заявке по номеру скана.</p>
      <UiButton variant="secondary" is-block @click="onClose">Закрыть</UiButton>
    </div>
    <form v-else class="suggest__form" @submit.prevent="onSubmit">
      <div class="suggest__photo">
        <img v-if="previewUrl !== null" :src="previewUrl" alt="Ваше фото этикетки" width="56" height="56">
        <span v-else class="suggest__photo-placeholder" aria-hidden="true">✓</span>
        <span class="suggest__photo-text">Фото уже приложено</span>
      </div>
      <label class="suggest__field">
        <span class="suggest__label">Название и производитель (необязательно)</span>
        <input
          class="suggest__input"
          type="text"
          name="comment"
          maxlength="200"
          autocomplete="off"
          placeholder="Например, Бюрнье Каберне 2022"
          :value="comment"
          @input="onInput"
        >
      </label>
      <UiButton type="submit" size="lg" is-block :is-loading="isSubmitting">Отправить</UiButton>
    </form>
  </UiBottomSheet>
</template>

<style scoped>
.suggest__form,
.suggest__done {
  display: flex;
  flex-direction: column;
  gap: var(--space-4);
}

.suggest__photo {
  display: flex;
  align-items: center;
  gap: var(--space-3);
}

.suggest__photo img {
  width: 56px;
  height: 56px;
  border-radius: var(--radius-sm);
  object-fit: cover;
}

.suggest__photo-placeholder {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 56px;
  height: 56px;
  border-radius: var(--radius-sm);
  background: var(--color-bg-tint-strong);
  color: var(--color-primary-dark);
}

.suggest__photo-text {
  color: var(--color-text-muted);
  font-size: var(--text-sm);
}

.suggest__field {
  display: flex;
  flex-direction: column;
  gap: var(--space-1);
}

.suggest__label {
  color: var(--color-text-muted);
  font-size: var(--text-sm);
}

.suggest__input {
  height: var(--control-height-lg);
  padding: 0 var(--space-4);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  background: var(--color-white);
  color: var(--color-text);
  font: inherit;
}

.suggest__input:focus {
  border-color: var(--color-primary);
  outline: none;
}

.suggest__done-title {
  font-family: var(--font-heading);
  font-size: var(--text-lg);
  color: var(--color-primary-dark);
}

.suggest__hint {
  color: var(--color-text-muted);
  font-size: var(--text-sm);
}
</style>
