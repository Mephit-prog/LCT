export interface SseMessage {
  event: string
  data: string
}

const parseBlock = (block: string): SseMessage | null => {
  const lines = block.split(/\r?\n/)
  let event = 'message'
  const dataLines: string[] = []
  for (const line of lines) {
    if (line.startsWith(':') || line.trim() === '') {
      continue
    }
    const separatorIndex = line.indexOf(':')
    const field = separatorIndex === -1 ? line : line.slice(0, separatorIndex)
    const rawValue = separatorIndex === -1 ? '' : line.slice(separatorIndex + 1)
    const value = rawValue.startsWith(' ') ? rawValue.slice(1) : rawValue
    if (field === 'event') {
      event = value
    } else if (field === 'data') {
      dataLines.push(value)
    }
  }
  if (dataLines.length === 0) {
    return null
  }
  return { event, data: dataLines.join('\n') }
}

export const parseSseChunk = (buffer: string): { messages: SseMessage[], rest: string } => {
  const normalized = buffer.replace(/\r\n/g, '\n')
  const blocks = normalized.split('\n\n')
  const rest = blocks.pop() ?? ''
  const messages: SseMessage[] = []
  for (const block of blocks) {
    const message = parseBlock(block)
    if (message !== null) {
      messages.push(message)
    }
  }
  return { messages, rest }
}

export async function* readSseMessages (response: Response): AsyncGenerator<SseMessage> {
  if (response.body === null) {
    return
  }
  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  try {
    while (true) {
      const { value, done } = await reader.read()
      if (done) {
        break
      }
      buffer += decoder.decode(value, { stream: true })
      const { messages, rest } = parseSseChunk(buffer)
      buffer = rest
      for (const message of messages) {
        yield message
      }
    }
    const tail = parseBlock(buffer)
    if (tail !== null) {
      yield tail
    }
  } finally {
    reader.releaseLock()
  }
}
