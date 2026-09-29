import { isApiError } from '@/services/apis/errors'

type LogContext = Record<string, unknown>

const describe = (error: unknown): LogContext => {
  if (isApiError(error)) {
    return {
      name: error.name,
      code: error.code,
      status: error.status,
      serverCode: error.serverCode,
      scanId: error.scanId,
      message: error.message
    }
  }
  if (error instanceof Error) {
    return { name: error.name, message: error.message, stack: error.stack }
  }
  return { value: String(error) }
}

export const logError = (scope: string, error: unknown, context: LogContext = {}): void => {
  console.error(`[svoe-vino:${scope}]`, { ...describe(error), ...context })
}

export const logWarn = (scope: string, message: string, context: LogContext = {}): void => {
  console.warn(`[svoe-vino:${scope}] ${message}`, context)
}
