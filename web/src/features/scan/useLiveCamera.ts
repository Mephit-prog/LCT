import { computed, onBeforeUnmount, onMounted, shallowRef, watch, watchEffect, type ComputedRef, type Ref } from 'vue'
import { logError } from '@/services/analytics/logger'
import {
  blockLiveCamera,
  canStartLiveCamera,
  captureFrame,
  liveCameraConstraints,
  shouldFallbackToSystemCamera
} from './live-camera'

export type LiveCameraPhase = 'closed' | 'live' | 'preview'

const phase = shallowRef<LiveCameraPhase>('closed')
const previewUrl = shallowRef<string | null>(null)
const isReady = shallowRef(false)
const stream = shallowRef<MediaStream | null>(null)
const videoElement = shallowRef<HTMLVideoElement | null>(null)

let generation = 0
let capturedFile: File | null = null

const stopStream = (media: MediaStream | null): void => {
  if (media === null) {
    return
  }
  for (const track of media.getTracks()) {
    track.stop()
  }
}

const releaseStream = (): void => {
  const media = stream.value
  stream.value = null
  const video = videoElement.value
  if (video !== null) {
    video.srcObject = null
  }
  stopStream(media)
  isReady.value = false
}

const revokePreview = (): void => {
  if (previewUrl.value !== null) {
    URL.revokeObjectURL(previewUrl.value)
    previewUrl.value = null
  }
  capturedFile = null
}

const attachStream = (): void => {
  const video = videoElement.value
  const media = stream.value
  if (video === null || media === null || video.srcObject === media) {
    return
  }
  video.srcObject = media
  const played = video.play()
  if (played instanceof Promise) {
    void played.catch(() => undefined)
  }
  markVideoReady()
}

const pauseLivePreview = (): void => {
  if (phase.value !== 'live') {
    return
  }
  generation += 1
  releaseStream()
}

export const closeLiveCamera = (): void => {
  generation += 1
  releaseStream()
  revokePreview()
  phase.value = 'closed'
}

export const startLiveCamera = (): boolean => {
  if (!canStartLiveCamera()) {
    return false
  }
  const current = generation + 1
  generation = current
  releaseStream()
  revokePreview()
  phase.value = 'live'

  let pending: Promise<MediaStream>
  try {
    pending = navigator.mediaDevices.getUserMedia(liveCameraConstraints)
  } catch (error) {
    blockLiveCamera()
    logError('camera', error)
    phase.value = 'closed'
    return false
  }

  void pending.then((media) => {
    if (current !== generation || phase.value !== 'live') {
      stopStream(media)
      return
    }
    stream.value = media
    attachStream()
  }).catch((error: unknown) => {
    if (current !== generation) {
      return
    }
    if (shouldFallbackToSystemCamera(error)) {
      blockLiveCamera()
    }
    logError('camera', error)
    releaseStream()
    phase.value = 'closed'
  })

  return true
}

export const requestScannerCamera = (openSystemCamera: () => void): void => {
  const started = startLiveCamera()
  if (!started) {
    openSystemCamera()
  }
}

const showCapturedPreview = (file: File): void => {
  generation += 1
  releaseStream()
  revokePreview()
  capturedFile = file
  previewUrl.value = URL.createObjectURL(file)
  phase.value = 'preview'
}

export const captureStill = async (): Promise<void> => {
  const video = videoElement.value
  if (video === null || video.videoWidth === 0 || video.videoHeight === 0) {
    return
  }
  const file = await captureFrame(video)
  showCapturedPreview(file)
}

export const takeCapturedFile = (): File | null => {
  const file = capturedFile
  if (file === null) {
    return null
  }
  capturedFile = null
  generation += 1
  releaseStream()
  if (previewUrl.value !== null) {
    URL.revokeObjectURL(previewUrl.value)
    previewUrl.value = null
  }
  phase.value = 'closed'
  return file
}

export const bindVideoElement = (element: Element | null): void => {
  videoElement.value = element instanceof HTMLVideoElement ? element : null
}

export const markVideoReady = (): void => {
  const video = videoElement.value
  isReady.value = video !== null && video.videoWidth > 0 && video.videoHeight > 0
}

export interface LiveCameraApi {
  phase: Ref<LiveCameraPhase>
  previewUrl: Ref<string | null>
  isReady: Ref<boolean>
  isOpen: ComputedRef<boolean>
}

export const useLiveCamera = (): LiveCameraApi => {
  const onVisibility = (): void => {
    if (document.visibilityState === 'hidden') {
      pauseLivePreview()
      return
    }
    if (phase.value === 'live' && stream.value === null) {
      startLiveCamera()
    }
  }

  const onPageHide = (): void => {
    pauseLivePreview()
  }

  onMounted(() => {
    document.addEventListener('visibilitychange', onVisibility)
    window.addEventListener('pagehide', onPageHide)
  })

  onBeforeUnmount(() => {
    document.removeEventListener('visibilitychange', onVisibility)
    window.removeEventListener('pagehide', onPageHide)
    document.body.style.overflow = ''
    closeLiveCamera()
  })

  watch(phase, (value) => {
    document.body.style.overflow = value === 'closed' ? '' : 'hidden'
  })

  watchEffect(() => {
    attachStream()
  })

  const isOpen = computed(() => phase.value !== 'closed')

  return {
    phase,
    previewUrl,
    isReady,
    isOpen
  }
}
