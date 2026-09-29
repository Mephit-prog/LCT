import { describe, expect, it } from 'vitest'
import { mount } from '@vue/test-utils'
import { sampleWine } from '@/test/samples'
import WineAttributes from '../WineAttributes.vue'
import WineHero from '../WineHero.vue'

const fullWine = sampleWine()

const sparseWine = sampleWine({
  slug: 'inkerman-rkatsiteli',
  title: 'Инкерман Ркацители',
  manufacturer: { name: 'Инкерман', slug: 'inkerman' },
  region: { name: 'Крым. Севастополь', image: null },
  grapes: [{ name: 'Ркацители', image: null }],
  category: { name: 'Белое сухое', backgroundGradient: null },
  color: 'Белое',
  alcohol: null,
  temperature: null,
  dishes: [],
  publicRating: null,
  roskachestvoRating: null
})

const blendWine = sampleWine({
  slug: 'derbent-vino-di-caspico-krasnoe',
  title: 'Дербент Вино Di Caspico Красное',
  grapes: ['Каберне Совиньон', 'Мерло', 'Саперави', 'Мальбек'].map((name) => ({ name, image: null }))
})

const ratedWine = sampleWine({
  slug: 'massandra-muskat',
  title: 'Массандра Мускат',
  publicRating: 4.9,
  roskachestvoRating: { score: 88, year: 2025 }
})

describe('WineAttributes', () => {
  it('рендерит все пять ячеек для полной карточки', () => {
    const wrapper = mount(WineAttributes, { props: { wine: fullWine } })
    const cells = wrapper.findAll('.wine-attributes__cell')
    expect(cells.length).toBe(5)
    expect(wrapper.text()).toContain('16–18 °C')
    expect(wrapper.text()).toContain('13.5 %')
  })

  it('не рендерит ячейки с пустыми значениями', () => {
    const wrapper = mount(WineAttributes, { props: { wine: sparseWine } })
    const keys = wrapper.findAll('.wine-attributes__cell').map((cell) => cell.attributes('data-attribute'))
    expect(keys).toEqual(['region', 'grapes', 'category'])
    expect(wrapper.text()).not.toContain('Крепость')
    expect(wrapper.text()).not.toContain('Температура')
  })

  it('сворачивает бленд из более трёх сортов и раскрывает по кнопке', async () => {
    const wrapper = mount(WineAttributes, { props: { wine: blendWine } })
    expect(wrapper.text()).toContain('Бленд:')
    await wrapper.find('.wine-attributes__expand').trigger('click')
    expect(wrapper.text()).toContain('Каберне Совиньон, Мерло, Саперави, Мальбек')
  })
})

describe('WineHero', () => {
  it('скрывает рейтинги, когда их нет, и использует фон по умолчанию', () => {
    const wrapper = mount(WineHero, { props: { wine: sparseWine } })
    expect(wrapper.find('.wine-hero__ratings').exists()).toBe(false)
    expect(wrapper.attributes('style')).toContain('var(--color-bg-soft)')
  })

  it('показывает оценку с двумя знаками и плашку Роскачества', () => {
    const wrapper = mount(WineHero, { props: { wine: ratedWine } })
    expect(wrapper.text()).toContain('4.90')
    expect(wrapper.text()).toContain('Роскачество')
    expect(wrapper.text()).toContain('88 · 2025')
  })
})
