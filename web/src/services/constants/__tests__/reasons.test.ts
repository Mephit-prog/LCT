import { describe, expect, it } from 'vitest'
import { isReasonCode, localizeReason, localizeReasons, reasonCodes, reasonLabels } from '../reasons'

describe('словарь причин', () => {
  it('содержит все коды контракта §6.1', () => {
    expect([...reasonCodes].sort()).toEqual([
      'other_manufacturer',
      'same_category',
      'same_grape',
      'same_manufacturer',
      'same_region',
      'same_series',
      'similar_label'
    ])
  })

  it('локализует каждый код на русский', () => {
    for (const code of reasonCodes) {
      expect(reasonLabels[code]).toMatch(/[а-яё]/i)
    }
    expect(localizeReason('same_grape')).toBe('тот же сорт')
  })

  it('игнорирует неизвестные коды и дубликаты', () => {
    expect(isReasonCode('unknown_reason')).toBe(false)
    expect(localizeReason('unknown_reason')).toBeNull()
    expect(localizeReasons(['same_grape', 'unknown', 'same_grape', 'other_manufacturer'])).toEqual(['тот же сорт', 'другая винодельня'])
  })
})
