import { imageMessages } from '@/services/constants/messages'

export type ImagePrepareErrorCode = 'not_image' | 'too_large' | 'decode_failed' | 'encode_failed'

export interface ImagePrepareError extends Error {
  name: 'ImagePrepareError'
  code: ImagePrepareErrorCode
}

export const createImagePrepareError = (code: ImagePrepareErrorCode, cause?: unknown): ImagePrepareError => {
  const messages: Record<ImagePrepareErrorCode, string> = {
    not_image: imageMessages.notImage,
    too_large: imageMessages.tooLarge,
    decode_failed: imageMessages.cannotDecode,
    encode_failed: imageMessages.cannotDecode
  }
  const error = new Error(messages[code], { cause }) as ImagePrepareError
  error.name = 'ImagePrepareError'
  error.code = code
  return error
}

export const isImagePrepareError = (value: unknown): value is ImagePrepareError => {
  return value instanceof Error && value.name === 'ImagePrepareError'
}

const heicExtensions = ['.heic', '.heif']

export const isImageLike = (file: File): boolean => {
  if (file.type.startsWith('image/')) {
    return true
  }
  const lowerName = file.name.toLowerCase()
  return file.type === '' && heicExtensions.some((extension) => lowerName.endsWith(extension))
}

export const validateImageFile = (file: File, maxBytes: number): ImagePrepareErrorCode | null => {
  if (!isImageLike(file)) {
    return 'not_image'
  }
  if (file.size > maxBytes) {
    return 'too_large'
  }
  return null
}

export interface Dimensions {
  width: number
  height: number
}

export const computeTargetSize = (source: Dimensions, maxSide: number): Dimensions => {
  const longest = Math.max(source.width, source.height)
  if (longest <= 0) {
    return { width: 1, height: 1 }
  }
  if (longest <= maxSide) {
    return { width: Math.round(source.width), height: Math.round(source.height) }
  }
  const scale = maxSide / longest
  return {
    width: Math.max(1, Math.round(source.width * scale)),
    height: Math.max(1, Math.round(source.height * scale))
  }
}

export interface EncodePlan {
  maxSide: number
  quality: number
}

export interface EncodePlanOptions {
  minQuality: number
  qualityStep: number
  minSide: number
  sideFactor: number
}

export const defaultEncodePlanOptions: EncodePlanOptions = {
  minQuality: 0.5,
  qualityStep: 0.1,
  minSide: 640,
  sideFactor: 0.75
}

const roundQuality = (value: number): number => Math.round(value * 100) / 100

export const planNextAttempt = (
  current: EncodePlan,
  bytes: number,
  targetBytes: number,
  options: EncodePlanOptions = defaultEncodePlanOptions
): EncodePlan | null => {
  if (bytes <= targetBytes) {
    return null
  }
  const loweredQuality = roundQuality(current.quality - options.qualityStep)
  if (loweredQuality >= options.minQuality) {
    return { maxSide: current.maxSide, quality: loweredQuality }
  }
  const loweredSide = Math.round(current.maxSide * options.sideFactor)
  if (loweredSide < options.minSide) {
    return null
  }
  return { maxSide: loweredSide, quality: roundQuality(Math.max(options.minQuality, current.quality + options.qualityStep * 2)) }
}

type DecodedImage = ImageBitmap | HTMLImageElement

const decodeWithBitmap = async (file: Blob): Promise<ImageBitmap> => {
  return createImageBitmap(file, { imageOrientation: 'from-image' })
}

const decodeWithElement = (file: Blob): Promise<HTMLImageElement> => {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file)
    const image = new Image()
    image.decoding = 'async'
    let settled = false
    const finish = (ok: boolean): void => {
      if (settled) return
      settled = true
      clearTimeout(timer)
      image.onload = null
      image.onerror = null
      URL.revokeObjectURL(url)
      if (ok) resolve(image)
      else reject(createImagePrepareError('decode_failed'))
    }
    const timer = setTimeout(() => finish(false), 3000)
    image.onload = (): void => finish(true)
    image.onerror = (): void => finish(false)
    image.src = url
  })
}

const decodeImage = async (file: Blob): Promise<DecodedImage> => {
  if (typeof createImageBitmap === 'function') {
    try {
      return await decodeWithBitmap(file)
    } catch {
      return decodeWithElement(file)
    }
  }
  return decodeWithElement(file)
}

const dimensionsOf = (image: DecodedImage): Dimensions => {
  if (image instanceof HTMLImageElement) {
    return { width: image.naturalWidth, height: image.naturalHeight }
  }
  return { width: image.width, height: image.height }
}

const releaseImage = (image: DecodedImage): void => {
  if (!(image instanceof HTMLImageElement)) {
    image.close()
  }
}

type AnyCanvas = OffscreenCanvas | HTMLCanvasElement

const createCanvas = (size: Dimensions): AnyCanvas => {
  if (typeof OffscreenCanvas === 'function') {
    return new OffscreenCanvas(size.width, size.height)
  }
  const canvas = document.createElement('canvas')
  canvas.width = size.width
  canvas.height = size.height
  return canvas
}

const drawTo = (canvas: AnyCanvas, image: DecodedImage, size: Dimensions): void => {
  const context = canvas.getContext('2d') as OffscreenCanvasRenderingContext2D | CanvasRenderingContext2D | null
  if (context === null) {
    throw createImagePrepareError('encode_failed')
  }
  context.drawImage(image, 0, 0, size.width, size.height)
}

const encodeJpeg = (canvas: AnyCanvas, quality: number): Promise<Blob> => {
  if (canvas instanceof HTMLCanvasElement) {
    return new Promise((resolve, reject) => {
      canvas.toBlob((blob) => {
        if (blob === null) {
          reject(createImagePrepareError('encode_failed'))
          return
        }
        resolve(blob)
      }, 'image/jpeg', quality)
    })
  }
  return canvas.convertToBlob({ type: 'image/jpeg', quality })
}

export interface PrepareImageOptions {
  maxSide: number
  quality: number
  targetBytes: number
  maxInputBytes: number
}

export interface PreparedImage {
  blob: Blob
  width: number
  height: number
  quality: number
  originalBytes: number
  sentBytes: number
  resizeMs: number
}

export const prepareImage = async (file: File, options: PrepareImageOptions): Promise<PreparedImage> => {
  const validationError = validateImageFile(file, options.maxInputBytes)
  if (validationError !== null) {
    throw createImagePrepareError(validationError)
  }

  const startedAt = performance.now()
  let image: DecodedImage
  try {
    image = await decodeImage(file)
  } catch (error) {
    throw isImagePrepareError(error) ? error : createImagePrepareError('decode_failed', error)
  }

  const source = dimensionsOf(image)
  if (source.width === 0 || source.height === 0) {
    releaseImage(image)
    throw createImagePrepareError('decode_failed')
  }

  let plan: EncodePlan | null = { maxSide: options.maxSide, quality: options.quality }
  let best: { blob: Blob, size: Dimensions, quality: number } | null = null
  try {
    while (plan !== null) {
      const size = computeTargetSize(source, plan.maxSide)
      const canvas = createCanvas(size)
      drawTo(canvas, image, size)
      const blob = await encodeJpeg(canvas, plan.quality)
      best = { blob, size, quality: plan.quality }
      plan = planNextAttempt(plan, blob.size, options.targetBytes)
    }
  } catch (error) {
    throw isImagePrepareError(error) ? error : createImagePrepareError('encode_failed', error)
  } finally {
    releaseImage(image)
  }

  if (best === null) {
    throw createImagePrepareError('encode_failed')
  }

  return {
    blob: best.blob,
    width: best.size.width,
    height: best.size.height,
    quality: best.quality,
    originalBytes: file.size,
    sentBytes: best.blob.size,
    resizeMs: Math.round(performance.now() - startedAt)
  }
}
