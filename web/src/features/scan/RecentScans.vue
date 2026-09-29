<script setup lang="ts">
import { onMounted } from 'vue'
import { useRecentScans } from './useRecentScans'

const { recentScans, load } = useRecentScans()

onMounted(load)
</script>

<template>
  <section v-if="recentScans.length > 0" class="recent-scans" aria-labelledby="recent-scans-title">
    <h2 id="recent-scans-title" class="recent-scans__title">Последние сканы</h2>
    <ul class="recent-scans__list">
      <li v-for="item in recentScans" :key="item.slug">
        <RouterLink class="recent-scans__item" :to="{ name: 'wine', params: { slug: item.slug }, query: { scan: item.scanId } }">
          <svg class="recent-scans__icon" viewBox="0 0 16 36" aria-hidden="true">
            <path fill="var(--color-primary)" d="M6.2 4.6h3.6V9.4c1.7.5 2.9 1.7 2.9 3.2v12.6c0 2.1-2 3.5-4.7 3.5s-4.7-1.4-4.7-3.5V12.6c0-1.5 1.2-2.7 2.9-3.2V4.6z" />
            <rect x="5.9" y="0.6" width="4.2" height="4.4" rx="0.8" fill="var(--color-primary-dark)" />
            <rect x="3.5" y="16.4" width="9" height="6.4" rx="0.6" fill="var(--color-bg-soft-strong)" />
            <rect x="4.5" y="18.5" width="4.6" height="0.8" rx="0.3" fill="var(--color-primary)" />
          </svg>
          <span class="recent-scans__label">{{ item.title }}</span>
          <span class="recent-scans__arrow" aria-hidden="true">›</span>
        </RouterLink>
      </li>
    </ul>
  </section>
</template>

<style scoped>
.recent-scans {
  padding: 0 var(--space-4);
}

.recent-scans__title {
  margin-bottom: var(--space-2);
  font-size: var(--text-md);
  font-family: var(--font-body);
  font-weight: 600;
  color: var(--color-text-muted);
}

.recent-scans__list {
  display: flex;
  flex-direction: column;
  border-top: 1px solid var(--color-border-soft);
}

.recent-scans__item {
  display: flex;
  align-items: center;
  gap: var(--space-3);
  min-height: var(--tap-size);
  padding: var(--space-2) 0;
  border-bottom: 1px solid var(--color-border-soft);
  color: var(--color-text);
}

.recent-scans__icon {
  display: block;
  flex-shrink: 0;
  width: 16px;
  height: 36px;
}

.recent-scans__label {
  flex: 1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.recent-scans__arrow {
  color: var(--color-text-muted);
  font-size: var(--text-lg);
}
</style>
