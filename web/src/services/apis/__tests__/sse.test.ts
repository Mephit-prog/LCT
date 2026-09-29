import { describe, expect, it } from 'vitest'
import { parseSseChunk, readSseMessages } from '../sse'

describe('parseSseChunk', () => {
  it('разбирает завершённые блоки и оставляет хвост', () => {
    const { messages, rest } = parseSseChunk('event: token\ndata: {"text":"a"}\n\nevent: done\ndata: {}\n\nevent: tok')
    expect(messages).toEqual([
      { event: 'token', data: '{"text":"a"}' },
      { event: 'done', data: '{}' }
    ])
    expect(rest).toBe('event: tok')
  })

  it('поддерживает CRLF и комментарии', () => {
    const { messages } = parseSseChunk(': keep-alive\r\nevent: token\r\ndata: {"text":"b"}\r\n\r\n')
    expect(messages).toEqual([{ event: 'token', data: '{"text":"b"}' }])
  })
})

describe('readSseMessages', () => {
  it('читает поток из Response', async () => {
    const encoder = new TextEncoder()
    const stream = new ReadableStream<Uint8Array>({
      start (controller) {
        controller.enqueue(encoder.encode('event: token\ndata: {"text":"Под"}\n\nevent: token\ndata: {"te'))
        controller.enqueue(encoder.encode('xt":"ойдёт"}\n\nevent: done\ndata: {}\n\n'))
        controller.close()
      }
    })
    const response = new Response(stream, { headers: { 'Content-Type': 'text/event-stream' } })
    const events: string[] = []
    for await (const message of readSseMessages(response)) {
      events.push(`${message.event}:${message.data}`)
    }
    expect(events).toEqual(['token:{"text":"Под"}', 'token:{"text":"ойдёт"}', 'done:{}'])
  })
})
