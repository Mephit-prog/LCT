export type ApiErrorCode =
  | 'NETWORK_OFFLINE'
  | 'NETWORK_FAILED'
  | 'TIMEOUT'
  | 'ABORTED'
  | 'SERVER_ERROR'
  | 'INVALID_RESPONSE'
  | 'IMAGE_TOO_LARGE'
  | 'UNSUPPORTED_IMAGE'
  | 'NOT_FOUND'
  | 'UNKNOWN'

export interface ApiError extends Error {
  name: 'ApiError'
  code: ApiErrorCode
  status: number | null
  serverCode: string | null
  scanId: string | null
}

interface ApiErrorOptions {
  status?: number | null
  serverCode?: string | null
  scanId?: string | null
  cause?: unknown
}

export const createApiError = (code: ApiErrorCode, message: string, options: ApiErrorOptions = {}): ApiError => {
  const error = new Error(message, { cause: options.cause }) as ApiError
  error.name = 'ApiError'
  error.code = code
  error.status = options.status ?? null
  error.serverCode = options.serverCode ?? null
  error.scanId = options.scanId ?? null
  return error
}

export const isApiError = (value: unknown): value is ApiError => {
  return value instanceof Error && value.name === 'ApiError' && 'code' in value
}

export const isAbortError = (value: unknown): boolean => {
  if (isApiError(value)) {
    return value.code === 'ABORTED'
  }
  return value instanceof Error && value.name === 'AbortError'
}

const reasonCodes: Partial<Record<string, ApiErrorCode>> = {
  invalid_size: 'IMAGE_TOO_LARGE',
  invalid_body_size: 'IMAGE_TOO_LARGE',
  not_found: 'NOT_FOUND',
  deadline: 'TIMEOUT',
  busy: 'SERVER_ERROR'
}

export const codeFromPayload = (status: number, reason: string | null): ApiErrorCode => {
  if (reason !== null) {
    const mapped = reasonCodes[reason]
    if (mapped !== undefined) {
      return mapped
    }
  }
  return codeFromStatus(status)
}

export const codeFromStatus = (status: number): ApiErrorCode => {
  if (status === 404) {
    return 'NOT_FOUND'
  }
  if (status === 413) {
    return 'IMAGE_TOO_LARGE'
  }
  if (status === 415 || status === 422) {
    return 'UNSUPPORTED_IMAGE'
  }
  if (status >= 500) {
    return 'SERVER_ERROR'
  }
  return 'UNKNOWN'
}

export const toApiError = (value: unknown, fallbackMessage: string): ApiError => {
  if (isApiError(value)) {
    return value
  }
  if (value instanceof Error && value.name === 'AbortError') {
    return createApiError('ABORTED', 'Запрос отменён', { cause: value })
  }
  if (value instanceof TypeError) {
    return createApiError('NETWORK_FAILED', fallbackMessage, { cause: value })
  }
  return createApiError('UNKNOWN', fallbackMessage, { cause: value })
}
