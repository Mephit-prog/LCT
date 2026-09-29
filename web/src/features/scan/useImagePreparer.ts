import { readonly, ref, type Ref } from 'vue'
import { env } from '@/services/constants/env'
import { maxImageBytesBeforeCompression } from '@/services/constants/timings'
import { prepareImage, type PrepareImageOptions, type PreparedImage } from './image-processing'

export interface UseImagePreparer {
  isPreparing: Readonly<Ref<boolean>>
  prepare: (file: File, overrides?: Partial<PrepareImageOptions>) => Promise<PreparedImage>
}

export const defaultPrepareOptions = (): PrepareImageOptions => ({
  maxSide: env.imageMaxSide,
  quality: env.imageQuality,
  targetBytes: env.imageTargetBytes,
  maxInputBytes: maxImageBytesBeforeCompression
})

export const useImagePreparer = (): UseImagePreparer => {
  const isPreparing = ref(false)

  const prepare = async (file: File, overrides: Partial<PrepareImageOptions> = {}): Promise<PreparedImage> => {
    isPreparing.value = true
    try {
      return await prepareImage(file, { ...defaultPrepareOptions(), ...overrides })
    } finally {
      isPreparing.value = false
    }
  }

  return { isPreparing: readonly(isPreparing), prepare }
}
