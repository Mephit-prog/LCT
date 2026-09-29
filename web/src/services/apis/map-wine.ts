import type { Wine, WineCard, WineSummary } from '@/services/types/api-types'

export const bottlePlaceholderUrl = '/bottles/placeholder.svg'
export const whiteBottlePlaceholderUrl = '/bottles/placeholder-white.svg'

const placeholderFor = (color: string): string => {
  if (color.toLocaleLowerCase('ru') === 'белое') {
    return whiteBottlePlaceholderUrl
  }
  return bottlePlaceholderUrl
}

const sweetnessLabels: Record<string, string> = {
  suhoe: 'сухое',
  polusuhoe: 'полусухое',
  polusladkoe: 'полусладкое',
  sladkoe: 'сладкое',
  bryut: 'брют'
}

const text = (value: string | null | undefined): string => {
  return value?.trim() ?? ''
}

const splitGrapes = (grape: string): string[] => {
  return grape.split(/[,;]/).map((part) => part.trim()).filter((part) => part !== '')
}

const withSweetness = (category: string, sweetness: string): string => {
  const label = sweetnessLabels[sweetness] ?? ''
  if (label === '') {
    return category
  }
  if (category === '') {
    return label.charAt(0).toUpperCase() + label.slice(1)
  }
  if (category.toLocaleLowerCase('ru').includes(label)) {
    return category
  }
  return `${category} ${label}`
}

export const mapWineCard = (card: WineCard, similar: WineSummary[] = []): Wine => {
  const name = text(card.name)
  const title = name === '' ? card.slug : name
  const media = text(card.media_url)
  const color = text(card.color)
  return {
    slug: card.slug,
    title,
    image: {
      url: media === '' ? placeholderFor(color) : media,
      altText: name === '' ? null : name
    },
    manufacturer: { name: text(card.producer), slug: '' },
    region: { name: '', image: null },
    grapes: splitGrapes(text(card.grape)).map((grape) => ({ name: grape, image: null })),
    category: {
      name: withSweetness(text(card.category), text(card.sweetness)),
      backgroundGradient: null
    },
    color: color === '' ? null : color,
    alcohol: null,
    temperature: null,
    description: null,
    dishes: [],
    publicRating: null,
    roskachestvoRating: null,
    similar
  }
}

export const toWineSummary = (card: WineCard): WineSummary => {
  const wine = mapWineCard(card)
  return {
    slug: wine.slug,
    title: wine.title,
    image: wine.image,
    manufacturer: wine.manufacturer.name,
    region: wine.region.name,
    category: wine.category.name,
    color: wine.color,
    publicRating: wine.publicRating
  }
}
