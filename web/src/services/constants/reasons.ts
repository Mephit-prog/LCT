export const reasonCodes = [
  'same_manufacturer',
  'same_grape',
  'same_category',
  'same_region',
  'similar_label',
  'other_manufacturer',
  'same_series'
] as const

export type ReasonCode = (typeof reasonCodes)[number]

export const reasonLabels: Record<ReasonCode, string> = {
  same_manufacturer: 'та же винодельня',
  same_grape: 'тот же сорт',
  same_category: 'та же категория',
  same_region: 'тот же регион',
  similar_label: 'похожая этикетка',
  other_manufacturer: 'другая винодельня',
  same_series: 'та же серия'
}

export const isReasonCode = (value: string): value is ReasonCode => {
  return (reasonCodes as readonly string[]).includes(value)
}

export const localizeReason = (code: string): string | null => {
  if (!isReasonCode(code)) {
    return null
  }
  return reasonLabels[code]
}

export const localizeReasons = (codes: readonly string[]): string[] => {
  const labels: string[] = []
  for (const code of codes) {
    const label = localizeReason(code)
    if (label === null || labels.includes(label)) {
      continue
    }
    labels.push(label)
  }
  return labels
}
