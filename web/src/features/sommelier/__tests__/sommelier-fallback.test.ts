import { describe, expect, it } from 'vitest'
import { sampleSummary, sampleWine } from '@/test/samples'
import { buildFallbackReply, offerDishAlternative, offerTasteAlternative, suggestDishMatches, suggestTasteMatches } from '../sommelier-fallback'

const redWine = sampleWine({
  similar: [
    sampleSummary('riesling', 'Рислинг', 'Белое'),
    sampleSummary('rose', 'Розе', 'Розовое')
  ]
})

const plainWine = sampleWine({
  title: 'Каберне',
  dishes: [],
  temperature: null,
  similar: [],
  publicRating: null,
  roskachestvoRating: null
})

describe('buildFallbackReply', () => {
  it('подтверждает вино, когда блюдо есть в справочнике dishes карточки', () => {
    const reply = buildFallbackReply({ mode: 'pairing', wine: redWine, candidates: [] }, { occasion: 'dinner', dish: 'meat' })
    expect(reply.text.startsWith('Подойдёт:')).toBe(true)
    expect(reply.text).toContain('16–18 °C')
    expect(reply.temperature).toBe('16–18')
    expect(reply.recommendations.length).toBeLessThanOrEqual(3)
  })

  it('советует взять другое, если цвет вина не подходит к блюду', () => {
    const reply = buildFallbackReply({ mode: 'pairing', wine: redWine, candidates: [] }, { occasion: 'dinner', dish: 'fish' })
    expect(reply.text.startsWith('Лучше взять другое:')).toBe(true)
    expect(reply.recommendations.every((item) => item.wine.color !== 'Красное')).toBe(true)
  })

  it('никогда не возвращает пустой текст даже при пустых dishes и temperature', () => {
    const reply = buildFallbackReply({ mode: 'pairing', wine: plainWine, candidates: [] }, { occasion: 'gift', dish: 'close' })
    expect(reply.text.length).toBeGreaterThan(20)
    expect(reply.temperature).toBeNull()
  })

  it('в режиме discovery рекомендует из кандидатов с учётом блюда', () => {
    const candidates = [
      sampleSummary('red', 'Саперави', 'Красное'),
      sampleSummary('white-a', 'Рислинг', 'Белое'),
      sampleSummary('white-b', 'Шардоне', 'Белое'),
      sampleSummary('white-c', 'Алиготе', 'Белое')
    ]
    const reply = buildFallbackReply({ mode: 'discovery', wine: null, candidates }, { occasion: 'dinner', dish: 'fish' })
    expect(reply.recommendations.length).toBe(3)
    expect(reply.recommendations.every((item) => item.wine.color !== 'Красное')).toBe(true)
    expect(reply.text).toContain('к рыбе')
  })

  it('не называет сухое вино подходящим, если выбран сладкий вкус', () => {
    const wine = sampleWine({
      title: 'АРАТТИ Рислинг сухое',
      category: { name: 'Белое сухое', backgroundGradient: null },
      color: 'Белое',
      similar: [
        sampleSummary('chardonnay', 'Шардоне сухое', 'Белое'),
        sampleSummary('muscat', 'Мускат полусладкое', 'Белое')
      ]
    })
    const reply = buildFallbackReply({ mode: 'pairing', wine, candidates: [] }, { occasion: 'try', dish: 'sweet' })
    expect(reply.text.startsWith('Лучше взять другое:')).toBe(true)
    expect(reply.text).toContain('сухое')
    expect(reply.text).toContain('сладкое')
    expect(reply.text).not.toContain('уверенный выбор')
    expect(reply.text).toContain('Вместо него возьмите Мускат полусладкое')
    expect(reply.temperature).toBeNull()
    expect(reply.recommendations.map((item) => item.wine.slug)).toEqual(['muscat'])
  })

  it('из каталога предлагает сладкое того же сорта, если среди похожих его нет', () => {
    const wine = sampleWine({
      title: 'АРАТТИ Рислинг сухое',
      grapes: [{ name: 'Рислинг', image: null }],
      category: { name: 'Белое сухое', backgroundGradient: null },
      color: 'Белое',
      similar: []
    })
    const pool = [
      sampleSummary('other-sweet', 'Мускат сладкое', 'Белое'),
      sampleSummary('riesling-sweet', 'Фанагория Рислинг полусладкое', 'Белое'),
      sampleSummary('riesling-dry', 'Рислинг сухое', 'Белое')
    ]
    const reply = buildFallbackReply({ mode: 'pairing', wine, candidates: [] }, { occasion: 'try', dish: 'sweet' })
    const suggestions = suggestTasteMatches(wine, 'sweet', pool)
    const offered = offerTasteAlternative(reply, suggestions, 'sweet')
    expect(suggestions.map((item) => item.slug)).toEqual(['riesling-sweet', 'other-sweet'])
    expect(offered.text).toContain('Вместо него возьмите Фанагория Рислинг полусладкое')
    expect(offered.recommendations[0]?.reason).toBe('сладкое')
  })

  it('к мясу не оставляет белые вина и подставляет красные', () => {
    const wine = sampleWine({
      title: 'Рислинг',
      color: 'Белое',
      category: { name: 'Белое сухое', backgroundGradient: null },
      dishes: [{ name: 'Рыба', image: null }],
      similar: [
        sampleSummary('chardonnay', 'Шардоне', 'Белое'),
        sampleSummary('riesling-2', 'Другой рислинг', 'белое')
      ]
    })
    const reply = buildFallbackReply({ mode: 'pairing', wine, candidates: [] }, { occasion: 'dinner', dish: 'meat' })
    const pool = [
      sampleSummary('white-hit', 'Алиготе', 'Белое'),
      sampleSummary('red-a', 'Саперави', 'Красное'),
      sampleSummary('red-b', 'Каберне', 'красное')
    ]
    const suggestions = suggestDishMatches(wine, 'meat', pool)
    const offered = offerDishAlternative(reply, suggestions, 'meat')
    expect(reply.text.startsWith('Лучше взять другое:')).toBe(true)
    expect(reply.recommendations).toEqual([])
    expect(reply.temperature).toBeNull()
    expect(suggestions.map((item) => item.slug)).toEqual(['red-a', 'red-b'])
    expect(offered.recommendations.map((item) => item.wine.slug)).toEqual(['red-a', 'red-b'])
    expect(offered.recommendations[0]?.reason).toBe('к мясу')
    expect(offered.text).toContain('Вместо него возьмите Саперави')
  })

  it('подтверждает полусладкое вино, если выбран сладкий вкус', () => {
    const wine = sampleWine({
      title: 'Мускат',
      category: { name: 'Белое полусладкое', backgroundGradient: null },
      color: 'Белое',
      similar: []
    })
    const reply = buildFallbackReply({ mode: 'pairing', wine, candidates: [] }, { occasion: 'try', dish: 'sweet' })
    expect(reply.text.startsWith('Подойдёт:')).toBe(true)
    expect(reply.text).toContain('полусладкое')
  })

  it('не превышает трёх предложений', () => {
    const wine = sampleWine({
      title: 'Пино Нуар',
      publicRating: 4.83,
      dishes: [{ name: 'Птица', image: null }, { name: 'Красное мясо', image: null }],
      temperature: '14–16'
    })
    const reply = buildFallbackReply({ mode: 'pairing', wine, candidates: [] }, { occasion: 'celebration', dish: 'romantic' })
    const sentences = reply.text.split(/[.!?]\s+/).filter((part) => part.trim() !== '')
    expect(sentences.length).toBeLessThanOrEqual(4)
  })
})
