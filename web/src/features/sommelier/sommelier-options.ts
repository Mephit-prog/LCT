export interface QuickOption {
  code: string
  label: string
}

export interface FollowUpQuestion {
  question: string
  options: QuickOption[]
}

export const occasionOptions: QuickOption[] = [
  { code: 'dinner', label: 'Ужин дома' },
  { code: 'gift', label: 'В подарок' },
  { code: 'celebration', label: 'Праздник' },
  { code: 'try', label: 'Просто попробовать' }
]

export const dishOptions: QuickOption[] = [
  { code: 'meat', label: 'Мясо' },
  { code: 'poultry', label: 'Птица' },
  { code: 'fish', label: 'Рыба' },
  { code: 'cheese', label: 'Сыры' },
  { code: 'vegetables', label: 'Овощи' },
  { code: 'dessert', label: 'Десерт' }
]

const followUps: Record<string, FollowUpQuestion> = {
  dinner: { question: 'Что на столе?', options: dishOptions },
  gift: {
    question: 'Кому подарок?',
    options: [
      { code: 'close', label: 'Близкому человеку' },
      { code: 'colleague', label: 'Коллеге' },
      { code: 'connoisseur', label: 'Ценителю вина' }
    ]
  },
  celebration: {
    question: 'Какой формат?',
    options: [
      { code: 'company', label: 'Большая компания' },
      { code: 'romantic', label: 'Романтический вечер' },
      { code: 'family', label: 'Семейный праздник' }
    ]
  },
  try: {
    question: 'Что вам ближе?',
    options: [
      { code: 'light', label: 'Лёгкое и свежее' },
      { code: 'rich', label: 'Насыщенное' },
      { code: 'sweet', label: 'Сладкое' }
    ]
  }
}

export const getFollowUp = (occasion: string): FollowUpQuestion | null => {
  return followUps[occasion] ?? null
}

export const labelOf = (options: QuickOption[], code: string | null): string | null => {
  if (code === null) {
    return null
  }
  return options.find((option) => option.code === code)?.label ?? null
}
