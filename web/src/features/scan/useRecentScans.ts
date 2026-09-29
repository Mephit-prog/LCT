import { computed, ref, type ComputedRef } from 'vue'
import { storageKeys } from '@/services/constants/storage-keys'
import { recentScansLimit } from '@/services/constants/timings'

export interface RecentScan {
  slug: string
  scanId: string
  title: string
  at: number
}

const isRecentScan = (value: unknown): value is RecentScan => {
  if (typeof value !== 'object' || value === null) {
    return false
  }
  const record = value as Record<string, unknown>
  return typeof record.slug === 'string' && typeof record.scanId === 'string' && typeof record.title === 'string' && typeof record.at === 'number'
}

const read = (): RecentScan[] => {
  try {
    const raw = localStorage.getItem(storageKeys.recentScans)
    if (raw === null) {
      return []
    }
    const parsed: unknown = JSON.parse(raw)
    if (!Array.isArray(parsed)) {
      return []
    }
    return parsed.filter(isRecentScan)
  } catch {
    return []
  }
}

const items = ref<RecentScan[]>([])
let isLoaded = false

const ensureLoaded = (): void => {
  if (isLoaded) {
    return
  }
  items.value = read()
  isLoaded = true
}

const persist = (): void => {
  try {
    localStorage.setItem(storageKeys.recentScans, JSON.stringify(items.value))
  } catch {
    return
  }
}

export interface UseRecentScans {
  recentScans: ComputedRef<RecentScan[]>
  remember: (entry: Omit<RecentScan, 'at'>) => void
  load: () => void
  clear: () => void
}

export const useRecentScans = (): UseRecentScans => {
  const remember = (entry: Omit<RecentScan, 'at'>): void => {
    ensureLoaded()
    const rest = items.value.filter((item) => item.slug !== entry.slug)
    items.value = [{ ...entry, at: Date.now() }, ...rest].slice(0, recentScansLimit)
    persist()
  }

  const clear = (): void => {
    items.value = []
    isLoaded = true
    persist()
  }

  return {
    recentScans: computed(() => items.value),
    remember,
    load: ensureLoaded,
    clear
  }
}
