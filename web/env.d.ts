/// <reference types="vite/client" />

declare module '*.vue' {
  import type { DefineComponent } from 'vue'
  const component: DefineComponent
  export default component
}

interface ImportMetaEnv {
  readonly VITE_API_BASE_URL: string
  readonly VITE_API_TOKEN: string
  readonly VITE_IMAGE_MAX_SIDE: string
  readonly VITE_IMAGE_QUALITY: string
  readonly VITE_IMAGE_TARGET_BYTES: string
  readonly VITE_SCAN_TIMEOUT_MS: string
  readonly VITE_SOMMELIER_TIMEOUT_MS: string
  readonly VITE_DEBUG_PANEL: string
  readonly VITE_PORTAL_BASE_URL: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
