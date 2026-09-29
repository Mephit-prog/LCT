<script setup lang="ts">
export type ScanTipIconName = 'frame' | 'light' | 'parallel'

interface ScanTipIconProps {
  name: ScanTipIconName
}

const { name } = defineProps<ScanTipIconProps>()
</script>

<template>
  <svg class="scan-tip-icon" viewBox="0 0 24 24" fill="none" aria-hidden="true">
    <g
      v-if="name === 'frame'"
      stroke="currentColor"
      stroke-width="1.7"
      stroke-linecap="round"
      stroke-linejoin="round"
    >
      <path d="M9 4.25H6.25A2.25 2.25 0 0 0 4 6.5V9" />
      <path d="M15 4.25h2.75A2.25 2.25 0 0 1 20 6.5V9" />
      <path d="M20 15v2.5a2.25 2.25 0 0 1-2.25 2.25H15" />
      <path d="M9 19.75H6.25A2.25 2.25 0 0 1 4 17.5V15" />
      <rect x="7.75" y="7.15" width="8.5" height="9.7" rx="1.2" />
      <path fill="currentColor" stroke="none" d="M12 9.15l.75.75-.75.75-.75-.75z" />
      <path d="M9.85 13.35h4.3" />
    </g>

    <g v-else-if="name === 'light'" fill="none" stroke="currentColor" stroke-linecap="round">
      <mask id="scan-tip-light-cut" maskUnits="userSpaceOnUse" x="0" y="0" width="24" height="24">
        <rect width="24" height="24" fill="white" />
        <path d="M8.35 15.65 15.65 8.35" stroke="black" stroke-width="2" stroke-linecap="round" />
      </mask>
      <circle cx="12" cy="12" r="4.85" fill="currentColor" stroke="none" mask="url(#scan-tip-light-cut)" />
      <path stroke-width="1.45" d="M12 2.55v3.15M12 18.3v3.15M2.55 12h3.15M18.3 12h3.15M5.35 5.35l2.25 2.25M16.4 16.4l2.25 2.25M18.65 5.35l-2.25 2.25M7.6 16.4l-2.25 2.25" />
    </g>

    <g
      v-else
      stroke="currentColor"
      stroke-width="1.7"
      stroke-linecap="round"
      stroke-linejoin="round"
    >
      <rect x="2.15" y="3" width="6.9" height="18" rx="1.45" />
      <path d="M4.2 5.15h2.8M4.55 18.7h2.1" />
      <path d="M15.55 3h2.2v5.5c1.15.7 2.55 1.85 2.55 3.3v6.9c0 1.45-1.55 2.3-3.65 2.3s-3.65-.85-3.65-2.3v-6.9c0-1.45 1.4-2.6 2.55-3.3V3z" />
      <path d="M14.35 14.15h4.5" />
    </g>
  </svg>
</template>

<style scoped>
.scan-tip-icon {
  display: block;
  width: 26px;
  height: 26px;
}
</style>
