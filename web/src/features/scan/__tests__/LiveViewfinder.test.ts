import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { mount, flushPromises, type VueWrapper } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { createMemoryHistory, createRouter } from 'vue-router'
import { storageKeys } from '@/services/constants/storage-keys'
import LiveViewfinder from '../LiveViewfinder.vue'
import { useScanStore } from '../scan.store'
import { closeLiveCamera, startLiveCamera } from '../useLiveCamera'

vi.mock('../live-camera', async () => {
  const actual = await vi.importActual<typeof import('../live-camera')>('../live-camera')
  return {
    ...actual,
    captureFrame: vi.fn(async () => new File([new Uint8Array([1, 2, 3])], 'label.jpg', { type: 'image/jpeg' }))
  }
})

const router = createRouter({
  history: createMemoryHistory(),
  routes: [{ path: '/', name: 'scan', component: { template: '<div />' } }]
})

const deferred = (): { promise: Promise<MediaStream>, resolve: (media: MediaStream) => void, reject: (error: unknown) => void } => {
  let resolveMedia: (media: MediaStream) => void = () => undefined
  let rejectMedia: (error: unknown) => void = () => undefined
  const promise = new Promise<MediaStream>((resolve, reject) => {
    resolveMedia = resolve
    rejectMedia = reject
  })
  return { promise, resolve: resolveMedia, reject: rejectMedia }
}

const fakeStream = (): { media: MediaStream, stop: ReturnType<typeof vi.fn> } => {
  const stop = vi.fn()
  const media = { getTracks: () => [{ stop }] } as unknown as MediaStream
  return { media, stop }
}

describe('LiveViewfinder', () => {
  let getUserMedia: ReturnType<typeof vi.fn>
  let wrapper: VueWrapper | null = null

  beforeEach(async () => {
    sessionStorage.clear()
    closeLiveCamera()
    const pinia = createPinia()
    setActivePinia(pinia)
    getUserMedia = vi.fn()
    Object.defineProperty(window, 'isSecureContext', { configurable: true, value: true })
    Object.defineProperty(navigator, 'mediaDevices', {
      configurable: true,
      value: { getUserMedia }
    })
    vi.spyOn(HTMLMediaElement.prototype, 'play').mockImplementation(() => Promise.resolve())
    vi.spyOn(URL, 'createObjectURL').mockReturnValue('blob:preview')
    vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => undefined)
    await router.push('/')
    await router.isReady()
    wrapper = mount(LiveViewfinder, {
      attachTo: document.body,
      global: { plugins: [pinia, router] }
    })
  })

  afterEach(() => {
    wrapper?.unmount()
    wrapper = null
    closeLiveCamera()
    vi.restoreAllMocks()
    document.body.style.overflow = ''
  })

  it('показывает живой кадр с рамкой', async () => {
    const pending = deferred()
    getUserMedia.mockReturnValue(pending.promise)
    expect(startLiveCamera()).toBe(true)
    await wrapper?.vm.$nextTick()

    const dialog = document.body.querySelector('[role="dialog"]')
    expect(dialog?.textContent).toContain('Этикетка целиком в рамке')
    expect(dialog?.querySelector('video')?.hasAttribute('playsinline')).toBe(true)
    expect(document.body.querySelector('[aria-label="Сфотографировать"]')).not.toBeNull()

    const { media } = fakeStream()
    pending.resolve(media)
    await flushPromises()
    expect(document.body.querySelector('video')).not.toBeNull()
  })

  it('после отказа закрывается и запоминает блокировку', async () => {
    const pending = deferred()
    getUserMedia.mockReturnValue(pending.promise)
    startLiveCamera()
    await wrapper?.vm.$nextTick()
    expect(document.body.querySelector('[role="dialog"]')).not.toBeNull()

    pending.reject(new DOMException('denied', 'NotAllowedError'))
    await flushPromises()
    await wrapper?.vm.$nextTick()

    expect(document.body.querySelector('[role="dialog"]')).toBeNull()
    expect(sessionStorage.getItem(storageKeys.liveCameraBlocked)).toBe('1')
  })

  it('отправляет весь кадр в скан', async () => {
    const pending = deferred()
    getUserMedia.mockReturnValue(pending.promise)
    const { media, stop } = fakeStream()
    startLiveCamera()
    await wrapper?.vm.$nextTick()
    pending.resolve(media)
    await flushPromises()
    await wrapper?.vm.$nextTick()

    const video = document.body.querySelector('video')
    expect(video).not.toBeNull()
    if (video === null) {
      return
    }
    Object.defineProperty(video, 'videoWidth', { configurable: true, value: 640 })
    Object.defineProperty(video, 'videoHeight', { configurable: true, value: 480 })
    video.dispatchEvent(new Event('loadedmetadata'))
    await wrapper?.vm.$nextTick()

    const shutter = document.body.querySelector<HTMLButtonElement>('[aria-label="Сфотографировать"]')
    expect(shutter?.disabled).toBe(false)
    shutter?.click()
    await flushPromises()
    await wrapper?.vm.$nextTick()

    expect(document.body.textContent).toContain('Переснять')
    const store = useScanStore()
    const start = vi.spyOn(store, 'start').mockResolvedValue(null)
    const send = Array.from(document.body.querySelectorAll('button')).find((button) => button.textContent?.includes('Отправить'))
    send?.click()
    await flushPromises()

    expect(start).toHaveBeenCalledOnce()
    const file = start.mock.calls[0]?.[0]
    expect(file).toBeInstanceOf(File)
    expect(file?.name).toBe('label.jpg')
    expect(start.mock.calls[0]?.[1]).toBe('camera')
    expect(document.body.querySelector('[role="dialog"]')).toBeNull()
    expect(stop).toHaveBeenCalled()
  })
})
