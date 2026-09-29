import { storageKeys } from '@/services/constants/storage-keys'
import type { ScanResult } from '@/services/types/api-types'
import { scanResultSchema } from '@/services/types/schemas'

const storedScansLimit = 12

const readAll = (): Record<string, unknown> => {
  try {
    const raw = sessionStorage.getItem(storageKeys.liveScans)
    if (raw === null) {
      return {}
    }
    const parsed: unknown = JSON.parse(raw)
    if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) {
      return {}
    }
    return parsed as Record<string, unknown>
  } catch {
    return {}
  }
}

export const readCachedScan = (scanId: string): ScanResult | null => {
  const parsed = scanResultSchema.safeParse(readAll()[scanId])
  if (!parsed.success) {
    return null
  }
  return parsed.data
}

export const writeCachedScan = (result: ScanResult): void => {
  try {
    const current = readAll()
    const next: Record<string, unknown> = { [result.scanId]: result }
    for (const key of Object.keys(current)) {
      if (key === result.scanId) {
        continue
      }
      if (Object.keys(next).length >= storedScansLimit) {
        break
      }
      next[key] = current[key]
    }
    sessionStorage.setItem(storageKeys.liveScans, JSON.stringify(next))
  } catch {
    return
  }
}
