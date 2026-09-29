import { describe, expect, it } from 'vitest'
import { mount } from '@vue/test-utils'
import { createMemoryHistory, createRouter } from 'vue-router'
import { sampleScan } from '@/test/samples'
import UncertainSheet from '../UncertainSheet.vue'

const createTestRouter = (): ReturnType<typeof createRouter> => createRouter({
  history: createMemoryHistory(),
  routes: [
    { path: '/', component: { template: '<div />' } },
    { path: '/wines/:slug', name: 'wine', component: { template: '<div />' } }
  ]
})

describe('UncertainSheet', () => {
  it('показывает до трёх кандидатов и закрывается по «Да, это оно»', async () => {
    const result = sampleScan('uncertain', 'scan-u')
    const wrapper = mount(UncertainSheet, {
      attachTo: document.body,
      props: { candidates: result.candidates, scanId: 'scan-u', modelValue: true },
      global: { plugins: [createTestRouter()] }
    })
    await wrapper.vm.$nextTick()

    const dialog = document.body.querySelector('[role="dialog"]')
    expect(dialog).not.toBeNull()
    expect(dialog?.textContent).toContain('Проверьте: это ваше вино?')
    expect(document.body.querySelectorAll('.uncertain__item').length).toBe(Math.min(3, result.candidates.length))
    expect(document.body.querySelector('.uncertain__item')?.getAttribute('href')).toContain('scan=scan-u')

    const confirm = Array.from(document.body.querySelectorAll('button')).find((button) => button.textContent?.includes('Да, это оно'))
    expect(confirm).toBeDefined()
    confirm?.click()
    await wrapper.vm.$nextTick()

    expect(wrapper.emitted('confirm')?.length).toBe(1)
    expect(wrapper.emitted('update:modelValue')?.[0]).toEqual([false])
    wrapper.unmount()
  })

  it('не является модальной: без оверлея и aria-modal', async () => {
    const result = sampleScan('uncertain', 'scan-u2')
    const wrapper = mount(UncertainSheet, {
      attachTo: document.body,
      props: { candidates: result.candidates, scanId: 'scan-u2', modelValue: true },
      global: { plugins: [createTestRouter()] }
    })
    await wrapper.vm.$nextTick()
    expect(document.body.querySelector('.ui-sheet__overlay')).toBeNull()
    expect(document.body.querySelector('[role="dialog"]')?.getAttribute('aria-modal')).toBeNull()
    wrapper.unmount()
  })
})
