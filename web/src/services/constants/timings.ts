export const processingPhases = {
  searchingFromMs: 1000,
  almostReadyFromMs: 2500,
  slowNoticeFromMs: 3000
} as const

export const uncertainSheetDelayMs = 600

export const toastDurationMs = 4000

export const typingIntervalMs = 18

export const maxImageBytesBeforeCompression = 25 * 1024 * 1024

export const retryImageMaxSide = 960

export const recentScansLimit = 5

export const liveScanTimeoutMs = 60_000
