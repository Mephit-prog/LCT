import { env } from '@/services/constants/env'
import { errorMessages } from '@/services/constants/messages'
import { apiErrorBodySchema, backendErrorBodySchema } from '@/services/types/schemas'
import { codeFromPayload, createApiError, isApiError, type ApiError } from './errors'

export interface LiveRequestOptions {
  method?: 'GET' | 'POST'
  data?: BodyInit | Record<string, unknown> | null
  headers?: Record<string, string>
  signal?: AbortSignal
  timeoutMs?: number
  scanId?: string | null
}

export interface LiveResponse {
  status: number
  data: unknown
}

interface Deadline {
  signal: AbortSignal
  cleanup: () => void
  didTimeOut: () => boolean
}

const isCanceledError = (error: unknown): boolean => {
  return typeof error === 'object' && error !== null && 'code' in error && error.code === 'ERR_CANCELED'
}

const isOffline = (): boolean => {
  return typeof navigator !== 'undefined' && navigator.onLine === false
}

const buildUrl = (path: string): string => {
  const base = env.apiBaseUrl.replace(/\/+$/, '')
  const suffix = path.startsWith('/') ? path : `/${path}`
  return `${base}${suffix}`
}

export const parseErrorPayload = (data: unknown): { code: string | null, message: string | null } => {
  const frontend = apiErrorBodySchema.safeParse(data)
  if (frontend.success) {
    return { code: frontend.data.error.code, message: frontend.data.error.message ?? null }
  }
  const backend = backendErrorBodySchema.safeParse(data)
  if (backend.success) {
    return { code: backend.data.reason_codes[0] ?? null, message: null }
  }
  return { code: null, message: null }
}

export const failureFromPayload = (status: number, data: unknown, scanId: string | null = null): ApiError => {
  const body = parseErrorPayload(data)
  const code = codeFromPayload(status, body.code)
  return createApiError(code, errorMessages[code], {
    status,
    serverCode: body.code,
    scanId
  })
}

const openDeadline = (external: AbortSignal | undefined, timeoutMs: number | undefined): Deadline => {
  const controller = new AbortController()
  let isTimedOut = false
  const onExternalAbort = (): void => {
    controller.abort()
  }
  external?.addEventListener('abort', onExternalAbort, { once: true })
  if (external?.aborted) {
    controller.abort()
  }
  const timer = timeoutMs === undefined
    ? null
    : setTimeout(() => {
      isTimedOut = true
      controller.abort()
    }, timeoutMs)
  return {
    signal: controller.signal,
    cleanup: (): void => {
      if (timer !== null) {
        clearTimeout(timer)
      }
      external?.removeEventListener('abort', onExternalAbort)
    },
    didTimeOut: (): boolean => isTimedOut
  }
}

const throwTransportFailure = (error: unknown, deadline: Deadline, scanId: string | null): never => {
  if (isApiError(error)) {
    throw error
  }
  if (deadline.didTimeOut()) {
    throw createApiError('TIMEOUT', errorMessages.TIMEOUT, { cause: error, scanId })
  }
  if (deadline.signal.aborted || isCanceledError(error)) {
    throw createApiError('ABORTED', errorMessages.ABORTED, { cause: error, scanId })
  }
  if (isOffline()) {
    throw createApiError('NETWORK_OFFLINE', errorMessages.NETWORK_OFFLINE, { cause: error })
  }
  throw createApiError('NETWORK_FAILED', errorMessages.NETWORK_FAILED, { cause: error, scanId })
}

export const requestLive = async (path: string, options: LiveRequestOptions = {}): Promise<LiveResponse> => {
  if (isOffline()) {
    throw createApiError('NETWORK_OFFLINE', errorMessages.NETWORK_OFFLINE, { scanId: options.scanId ?? null })
  }

  const headers: Record<string, string> = { ...(options.headers ?? {}) }
  if (env.apiToken !== '') {
    headers.Authorization = `Bearer ${env.apiToken}`
  }
  if (typeof FormData !== 'undefined' && options.data instanceof FormData) {
    delete headers['Content-Type']
  }

  const deadline = openDeadline(options.signal, options.timeoutMs)
  let response: LiveResponse
  try {
    const { default: axios } = await import('axios')
    const result = await axios.request<unknown>({
      url: buildUrl(path),
      method: options.method ?? 'GET',
      data: options.data ?? undefined,
      headers,
      signal: deadline.signal,
      validateStatus: () => true
    })
    response = { status: result.status, data: result.data }
  } catch (error) {
    deadline.cleanup()
    return throwTransportFailure(error, deadline, options.scanId ?? null)
  }
  deadline.cleanup()

  if (response.status < 200 || response.status >= 300) {
    throw failureFromPayload(response.status, response.data, options.scanId ?? null)
  }
  return response
}
