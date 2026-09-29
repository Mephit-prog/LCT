<template>
  <div class="sticky-bar">
    <div class="sticky-bar__inner">
      <slot />
    </div>
  </div>
</template>

<style scoped>
.sticky-bar {
  position: fixed;
  left: 0;
  right: 0;
  bottom: 0;
  z-index: 40;
  padding: var(--space-3) var(--space-4) calc(var(--space-3) + var(--safe-bottom));
  background: rgb(254 253 250 / 92%);
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  border-top: 1px solid var(--color-border-soft);
}

.sticky-bar__inner {
  display: flex;
  gap: var(--space-3);
  width: 100%;
  max-width: var(--container-max);
  margin: 0 auto;
}

.sticky-bar__inner > :deep(*) {
  flex: 1;
}

.sticky-bar__inner > :deep(.sticky-bar__aux) {
  flex: 0 0 auto;
}
</style>
