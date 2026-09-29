import { describe, expect, it } from 'vitest'
import { bottlePlaceholderUrl, mapWineCard, whiteBottlePlaceholderUrl } from '@/services/apis/map-wine'
import type { WineCard } from '@/services/types/api-types'
import { wineSchema } from '@/services/types/schemas'

const card = (overrides: Partial<WineCard> = {}): WineCard => ({
  slug: 'fanagoriya-risling',
  name: 'Авторский стиль. Рислинг',
  producer: 'Фанагория',
  color: 'Белое',
  vintage: '2021',
  grape: 'рислинг, шардоне',
  category: 'Белое',
  sweetness: 'suhoe',
  media_url: '/api/media/fanagoriya.webp',
  ...overrides
})

describe('mapWineCard', () => {
  it('переносит поля карточки бэкенда и добавляет сладость', () => {
    const wine = mapWineCard(card())
    expect(wineSchema.safeParse(wine).success).toBe(true)
    expect(wine.title).toBe('Авторский стиль. Рислинг')
    expect(wine.manufacturer).toEqual({ name: 'Фанагория', slug: '' })
    expect(wine.grapes.map((grape) => grape.name)).toEqual(['рислинг', 'шардоне'])
    expect(wine.category).toEqual({ name: 'Белое сухое', backgroundGradient: null })
    expect(wine.color).toBe('Белое')
    expect(wine.image.url).toBe('/api/media/fanagoriya.webp')
    expect(wine.region.name).toBe('')
    expect(wine.description).toBeNull()
    expect(wine.alcohol).toBeNull()
    expect(wine.temperature).toBeNull()
    expect(wine.dishes).toEqual([])
    expect(wine.publicRating).toBeNull()
    expect(wine.roskachestvoRating).toBeNull()
    expect(wine.similar).toEqual([])
  })

  it('не дублирует сладость и подставляет плейсхолдер без фото', () => {
    const wine = mapWineCard(card({
      category: 'Белое сухое',
      sweetness: 'suhoe',
      media_url: null,
      color: '  ',
      grape: '',
      name: '  '
    }))
    expect(wine.title).toBe('fanagoriya-risling')
    expect(wine.category.name).toBe('Белое сухое')
    expect(wine.image.url).toBe(bottlePlaceholderUrl)
    expect(wine.image.altText).toBeNull()
    expect(wine.color).toBeNull()
    expect(wine.grapes).toEqual([])
  })

  it('без фото белому вину ставит зелёную бутылку, красному — бордовую', () => {
    const white = mapWineCard(card({ media_url: '', color: 'Белое' }))
    const red = mapWineCard(card({ media_url: null, color: 'Красное' }))
    expect(white.image.url).toBe(whiteBottlePlaceholderUrl)
    expect(red.image.url).toBe(bottlePlaceholderUrl)
  })

  it('берёт сладость как категорию, если категории нет', () => {
    const wine = mapWineCard(card({ category: '', sweetness: 'bryut' }))
    expect(wine.category.name).toBe('Брют')
  })
})
