const parseNumber = (raw: string | undefined, fallback: number): number => {
  if (raw === undefined || raw === '') {
    return fallback
  }
  const value = Number(raw)
  if (!Number.isFinite(value)) {
    return fallback
  }
  return value
}

const parseBoolean = (raw: string | undefined, fallback: boolean): boolean => {
  if (raw === undefined || raw === '') {
    return fallback
  }
  return raw === 'true' || raw === '1'
}

export interface AppEnv {
  apiBaseUrl: string
  apiToken: string
  imageMaxSide: number
  imageQuality: number
  imageTargetBytes: number
  scanTimeoutMs: number
  sommelierTimeoutMs: number
  isDebugPanelForced: boolean
  portalBaseUrl: string
}

export const env: AppEnv = {
  apiBaseUrl: import.meta.env.VITE_API_BASE_URL || '/api',
  apiToken: import.meta.env.VITE_API_TOKEN?.trim() ?? '',
  imageMaxSide: parseNumber(import.meta.env.VITE_IMAGE_MAX_SIDE, 1280),
  imageQuality: parseNumber(import.meta.env.VITE_IMAGE_QUALITY, 0.85),
  imageTargetBytes: parseNumber(import.meta.env.VITE_IMAGE_TARGET_BYTES, 512000),
  scanTimeoutMs: Math.max(70000, parseNumber(import.meta.env.VITE_SCAN_TIMEOUT_MS, 70000)),
  sommelierTimeoutMs: parseNumber(import.meta.env.VITE_SOMMELIER_TIMEOUT_MS, 6000),
  isDebugPanelForced: parseBoolean(import.meta.env.VITE_DEBUG_PANEL, false),
  portalBaseUrl: import.meta.env.VITE_PORTAL_BASE_URL || 'https://vino-svoe.ru'
}
