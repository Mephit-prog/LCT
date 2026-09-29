import { storageKeys } from '@/services/constants/storage-keys'

export const liveCameraConstraints: MediaStreamConstraints = {
  audio: false,
  video: {
    facingMode: { ideal: 'environment' }
  }
}

const fallbackErrorNames = new Set([
  'NotAllowedError',
  'NotFoundError',
  'SecurityError',
  'NotReadableError',
  'OverconstrainedError',
  'AbortError',
  'TypeError'
])

const captureQuality = 0.92

export const canUseLiveCamera = (): boolean => {
  if (typeof window === 'undefined' || typeof navigator === 'undefined') {
    return false
  }
  const getUserMedia = navigator.mediaDevices?.getUserMedia
  return window.isSecureContext && typeof getUserMedia === 'function'
}

export const isLiveCameraBlocked = (): boolean => {
  if (typeof sessionStorage === 'undefined') {
    return false
  }
  try {
    return sessionStorage.getItem(storageKeys.liveCameraBlocked) === '1'
  } catch {
    return false
  }
}

export const blockLiveCamera = (): void => {
  if (typeof sessionStorage === 'undefined') {
    return
  }
  try {
    sessionStorage.setItem(storageKeys.liveCameraBlocked, '1')
  } catch {
    return
  }
}

export const canStartLiveCamera = (): boolean => {
  return canUseLiveCamera() && !isLiveCameraBlocked()
}

export const shouldFallbackToSystemCamera = (error: unknown): boolean => {
  if (!(error instanceof Error)) {
    return true
  }
  if (fallbackErrorNames.has(error.name)) {
    return true
  }
  return error.name.length > 0
}

export const captureFrame = (video: HTMLVideoElement): Promise<File> => {
  const width = video.videoWidth
  const height = video.videoHeight
  if (width === 0 || height === 0) {
    return Promise.reject(new Error('empty-frame'))
  }
  const canvas = document.createElement('canvas')
  canvas.width = width
  canvas.height = height
  const context = canvas.getContext('2d')
  if (context === null) {
    return Promise.reject(new Error('no-canvas'))
  }
  context.drawImage(video, 0, 0, width, height)
  return new Promise((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (blob === null) {
        reject(new Error('encode-failed'))
        return
      }
      resolve(new File([blob], 'label.jpg', { type: 'image/jpeg' }))
    }, 'image/jpeg', captureQuality)
  })
}
