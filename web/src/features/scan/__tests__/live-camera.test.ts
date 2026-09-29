import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { storageKeys } from '@/services/constants/storage-keys'
import {
  blockLiveCamera,
  canStartLiveCamera,
  canUseLiveCamera,
  captureFrame,
  isLiveCameraBlocked,
  liveCameraConstraints,
  shouldFallbackToSystemCamera
} from '../live-camera'
import { closeLiveCamera, requestScannerCamera, startLiveCamera } from '../useLiveCamera'

const installCamera = (getUserMedia: ReturnType<typeof vi.fn> | undefined, isSecure = true): void => {
  Object.defineProperty(window, 'isSecureContext', { configurable: true, value: isSecure })
  Object.defineProperty(navigator, 'mediaDevices', {
    configurable: true,
    value: getUserMedia === undefined ? undefined : { getUserMedia }
  })
}

describe('shouldFallbackToSystemCamera', () => {
  it('переводит отказ, отсутствие камеры и прочие сбои на системную съёмку', () => {
    expect(shouldFallbackToSystemCamera(new DOMException('denied', 'NotAllowedError'))).toBe(true)
    expect(shouldFallbackToSystemCamera(new DOMException('missing', 'NotFoundError'))).toBe(true)
    expect(shouldFallbackToSystemCamera(new DOMException('insecure', 'SecurityError'))).toBe(true)
    expect(shouldFallbackToSystemCamera(new Error('boom'))).toBe(true)
    expect(shouldFallbackToSystemCamera('offline')).toBe(true)
  })
})

describe('canStartLiveCamera', () => {
  beforeEach(() => {
    sessionStorage.clear()
  })

  it('требует secure context и getUserMedia', () => {
    installCamera(vi.fn(), false)
    expect(canUseLiveCamera()).toBe(false)
    expect(canStartLiveCamera()).toBe(false)

    installCamera(undefined, true)
    expect(canUseLiveCamera()).toBe(false)

    installCamera(vi.fn(), true)
    expect(canStartLiveCamera()).toBe(true)
  })

  it('помнит отказ в sessionStorage', () => {
    installCamera(vi.fn(), true)
    expect(isLiveCameraBlocked()).toBe(false)
    blockLiveCamera()
    expect(sessionStorage.getItem(storageKeys.liveCameraBlocked)).toBe('1')
    expect(isLiveCameraBlocked()).toBe(true)
    expect(canStartLiveCamera()).toBe(false)
  })
})

describe('startLiveCamera', () => {
  beforeEach(() => {
    sessionStorage.clear()
    closeLiveCamera()
  })

  afterEach(() => {
    closeLiveCamera()
  })

  it('не вызывает getUserMedia, если камера уже недоступна', () => {
    const getUserMedia = vi.fn()
    installCamera(getUserMedia, false)
    const openSystem = vi.fn()
    requestScannerCamera(openSystem)
    expect(startLiveCamera()).toBe(false)
    expect(getUserMedia).not.toHaveBeenCalled()
    expect(openSystem).toHaveBeenCalledOnce()
  })

  it('после отказа следующий запуск не трогает getUserMedia', async () => {
    const getUserMedia = vi.fn().mockRejectedValue(new DOMException('denied', 'NotAllowedError'))
    installCamera(getUserMedia, true)
    const openSystem = vi.fn()
    expect(startLiveCamera()).toBe(true)
    expect(getUserMedia).toHaveBeenCalledTimes(1)
    expect(getUserMedia).toHaveBeenCalledWith(liveCameraConstraints)
    await vi.waitFor(() => {
      expect(isLiveCameraBlocked()).toBe(true)
    })
    requestScannerCamera(openSystem)
    expect(getUserMedia).toHaveBeenCalledOnce()
    expect(openSystem).toHaveBeenCalledOnce()
  })

  it('останавливает треки при закрытии', async () => {
    const track = { stop: vi.fn() }
    const media = { getTracks: () => [track] } as unknown as MediaStream
    const getUserMedia = vi.fn().mockResolvedValue(media)
    installCamera(getUserMedia, true)
    expect(startLiveCamera()).toBe(true)
    await vi.waitFor(() => expect(getUserMedia).toHaveBeenCalledTimes(1))
    closeLiveCamera()
    await vi.waitFor(() => expect(track.stop).toHaveBeenCalledOnce())
  })
})

describe('captureFrame', () => {
  it('отклоняет пустой кадр', async () => {
    const video = document.createElement('video')
    Object.defineProperty(video, 'videoWidth', { configurable: true, value: 0 })
    Object.defineProperty(video, 'videoHeight', { configurable: true, value: 0 })
    await expect(captureFrame(video)).rejects.toThrow('empty-frame')
  })

  it('сохраняет весь кадр как jpeg', async () => {
    const drawImage = vi.fn()
    const getContext = vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue({
      drawImage
    } as unknown as CanvasRenderingContext2D)
    const toBlob = vi.spyOn(HTMLCanvasElement.prototype, 'toBlob').mockImplementation((callback) => {
      callback?.(new Blob(['jpeg'], { type: 'image/jpeg' }))
    })
    const video = document.createElement('video')
    Object.defineProperty(video, 'videoWidth', { configurable: true, value: 1280 })
    Object.defineProperty(video, 'videoHeight', { configurable: true, value: 720 })

    const file = await captureFrame(video)

    expect(file.name).toBe('label.jpg')
    expect(file.type).toBe('image/jpeg')
    expect(drawImage).toHaveBeenCalledWith(video, 0, 0, 1280, 720)
    getContext.mockRestore()
    toBlob.mockRestore()
  })
})
