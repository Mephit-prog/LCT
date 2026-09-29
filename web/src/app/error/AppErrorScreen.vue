<script setup lang="ts">
import UiButton from '@/components/ui/UiButton.vue'
import AgeFooter from '@/components/layout/AgeFooter.vue'

const emit = defineEmits<{ recover: [] }>()

const onRecover = (): void => emit('recover')
</script>

<template>
  <main class="app-error">
    <div class="app-error__body">
      <h1 class="app-error__title">Что-то пошло не так</h1>
      <p class="app-error__text">Мы уже знаем об ошибке. Вернитесь на экран сканера и попробуйте ещё раз.</p>
      <UiButton size="lg" is-block @click="onRecover">На экран сканера</UiButton>
    </div>
    <AgeFooter />
  </main>
</template>

<style scoped>
.app-error {
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  min-height: 100dvh;
  max-width: var(--container-max);
  margin: 0 auto;
}

.app-error__body {
  display: flex;
  flex-direction: column;
  gap: var(--space-4);
  padding: var(--space-8) var(--space-4);
  padding-top: calc(var(--space-8) + var(--safe-top));
}

.app-error__title {
  color: var(--color-primary-dark);
}

.app-error__text {
  color: var(--color-text-muted);
}
</style>
