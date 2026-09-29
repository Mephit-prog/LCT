import { storageKeys } from '@/services/constants/storage-keys'

const readSet = (): Set<string> => {
  try {
    const raw = sessionStorage.getItem(storageKeys.uncertainConfirmed)
    if (raw === null) {
      return new Set()
    }
    const parsed: unknown = JSON.parse(raw)
    if (!Array.isArray(parsed)) {
      return new Set()
    }
    return new Set(parsed.filter((item): item is string => typeof item === 'string'))
  } catch {
    return new Set()
  }
}

export const isUncertainHandled = (scanId: string): boolean => {
  return readSet().has(scanId)
}

export const markUncertainHandled = (scanId: string): void => {
  try {
    const set = readSet()
    set.add(scanId)
    sessionStorage.setItem(storageKeys.uncertainConfirmed, JSON.stringify([...set].slice(-50)))
  } catch {
    return
  }
}
