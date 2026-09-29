import { describe, expect, it } from 'vitest'
import { apis } from '@/services/apis'
import { isApiError } from '@/services/apis/errors'

describe('клиентские заглушки', () => {
  it('suggest завершается без сети', async () => {
    await expect(apis.suggest('scan-live', { comment: 'Рислинг' })).resolves.toBeUndefined()
  })

  it('сомелье сразу заканчивает поток', async () => {
    const events = []
    for await (const event of apis.sommelier({
      mode: 'pairing',
      scanId: null,
      slug: 'fanagoria',
      answers: { occasion: 'dinner', dish: 'fish' }
    }, new AbortController().signal)) {
      events.push(event)
    }
    expect(events).toEqual([])
  })

  it('неизвестный scanId не запрашивает бэкенд', async () => {
    sessionStorage.clear()
    await expect(apis.getScan('missing-scan')).rejects.toSatisfy((error: unknown) => {
      return isApiError(error) && error.code === 'NOT_FOUND' && error.status === 404
    })
  })
})
