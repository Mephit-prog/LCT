<template>
  <footer class="age-footer">
    <span class="age-footer__badge" aria-hidden="true">18+</span>
    <p class="age-footer__text">Чрезмерное употребление алкоголя вредит вашему здоровью</p>
  </footer>
</template>

<style scoped>
.age-footer {
  display: flex;
  align-items: center;
  gap: var(--space-3);
  padding: var(--space-4);
  color: var(--color-text-muted);
  font-size: var(--text-xs);
  line-height: 1.4;
}

.age-footer__badge {
  flex-shrink: 0;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 36px;
  height: 36px;
  border: 2px solid var(--color-text-muted);
  border-radius: 50%;
  font-weight: 700;
  font-size: var(--text-xs);
}
</style>
