import { useToast } from '@/composables/useToast'
import { logError } from '@/services/analytics/logger'
import { uiMessages } from '@/services/constants/messages'

export interface ShareInput {
  title: string
  text?: string
  url: string
}

export interface UseShare {
  share: (input: ShareInput) => Promise<void>
}

const isAbort = (error: unknown): boolean => error instanceof Error && error.name === 'AbortError'

export const useShare = (): UseShare => {
  const toast = useToast()

  const copyLink = async (url: string): Promise<void> => {
    if (typeof navigator === 'undefined' || navigator.clipboard === undefined) {
      toast.error(uiMessages.shareFailed)
      return
    }
    try {
      await navigator.clipboard.writeText(url)
      toast.success(uiMessages.linkCopied)
    } catch (error) {
      logError('share', error)
      toast.error(uiMessages.shareFailed)
    }
  }

  const share = async (input: ShareInput): Promise<void> => {
    if (typeof navigator === 'undefined' || typeof navigator.share !== 'function') {
      await copyLink(input.url)
      return
    }
    try {
      await navigator.share(input)
    } catch (error) {
      if (isAbort(error)) {
        return
      }
      logError('share', error)
      await copyLink(input.url)
    }
  }

  return { share }
}
