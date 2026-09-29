import { describe, expect, it } from 'vitest'
import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { createMemoryHistory, createRouter } from 'vue-router'
import { useScanStore } from '@/features/scan/scan.store'
import { sampleScan } from '@/test/samples'
import SimilarPage from '../SimilarPage.vue'

const mountPage = async (scanId: string): Promise<ReturnType<typeof mount>> => {
  const router = createRouter({
    history: createMemoryHistory(),
    routes: [
      { path: '/scan', name: 'scan', component: { template: '<div />' } },
      { path: '/wines/:slug', name: 'wine', component: { template: '<div />' } },
      { path: '/scan/:scanId/similar', name: 'similar', component: SimilarPage }
    ]
  })
  await router.push({ name: 'similar', params: { scanId } })
  await router.isReady()
  const wrapper = mount(SimilarPage, { global: { plugins: [router] } })
  await flushPromises()
  return wrapper
}

describe('SimilarPage', () => {
  it('скрывает блок альтернатив, когда массив пуст, и показывает похожие с причинами', async () => {
    const pinia = createPinia()
    setActivePinia(pinia)
    const store = useScanStore()
    const result = sampleScan('not_found', 'scan-empty-alt')
    store.rememberResult({ ...result, alternatives: [] })

    const wrapper = await mountPage('scan-empty-alt')

    expect(wrapper.text()).toContain('Точного совпадения в каталоге нет')
    expect(wrapper.text()).toContain('Похожие вина')
    expect(wrapper.text()).not.toContain('Похожие вина других виноделен')
    expect(wrapper.findAll('.summary-card--list').length).toBe(Math.min(5, result.similar.length))
    expect(wrapper.text()).toContain('похожая этикетка')
    expect(wrapper.text()).toContain('Что мы распознали')
    expect(wrapper.text()).toContain('Предложение вина и цифровой сомелье пока недоступны')
    expect(wrapper.find('button').text()).not.toContain('Предложить вино')
    wrapper.unmount()
  })

  it('рендерит блок альтернатив при непустом массиве', async () => {
    setActivePinia(createPinia())
    const store = useScanStore()
    store.rememberResult(sampleScan('not_found', 'scan-with-alt'))

    const wrapper = await mountPage('scan-with-alt')
    expect(wrapper.text()).toContain('Похожие вина других виноделен')
    expect(wrapper.text()).toContain('другая винодельня')
    wrapper.unmount()
  })
})
