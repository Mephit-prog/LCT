import { readonly, ref, type Ref } from 'vue'
import { toastDurationMs } from '@/services/constants/timings'

export type ToastTone = 'neutral' | 'success' | 'error'

export interface ToastItem {
  id: number
  message: string
  tone: ToastTone
  actionLabel: string | null
  onAction: (() => void) | null
}

interface ToastOptions {
  tone?: ToastTone
  actionLabel?: string
  onAction?: () => void
  durationMs?: number
}

const toasts: Ref<ToastItem[]> = ref([])
const timers = new Map<number, ReturnType<typeof setTimeout>>()
let nextId = 1

const dismiss = (id: number): void => {
  const timer = timers.get(id)
  if (timer !== undefined) {
    clearTimeout(timer)
    timers.delete(id)
  }
  toasts.value = toasts.value.filter((toast) => toast.id !== id)
}

const show = (message: string, options: ToastOptions = {}): number => {
  const id = nextId
  nextId += 1
  toasts.value = [...toasts.value.slice(-2), {
    id,
    message,
    tone: options.tone ?? 'neutral',
    actionLabel: options.actionLabel ?? null,
    onAction: options.onAction ?? null
  }]
  timers.set(id, setTimeout(() => dismiss(id), options.durationMs ?? toastDurationMs))
  return id
}

export interface UseToast {
  toasts: Readonly<Ref<readonly ToastItem[]>>
  show: (message: string, options?: ToastOptions) => number
  error: (message: string, options?: ToastOptions) => number
  success: (message: string, options?: ToastOptions) => number
  dismiss: (id: number) => void
}

export const useToast = (): UseToast => ({
  toasts: readonly(toasts),
  show,
  error: (message, options = {}) => show(message, { ...options, tone: 'error' }),
  success: (message, options = {}) => show(message, { ...options, tone: 'success' }),
  dismiss
})
