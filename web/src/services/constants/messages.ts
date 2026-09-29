import type { ApiErrorCode } from '@/services/apis/errors'

export const errorMessages: Record<ApiErrorCode, string> = {
  NETWORK_OFFLINE: 'Нет подключения к интернету',
  NETWORK_FAILED: 'Не удалось отправить фото',
  TIMEOUT: 'Сервис отвечает слишком долго',
  ABORTED: 'Запрос отменён',
  SERVER_ERROR: 'Сервис временно недоступен',
  INVALID_RESPONSE: 'Сервис временно недоступен',
  IMAGE_TOO_LARGE: 'Фото слишком большое',
  UNSUPPORTED_IMAGE: 'Не удалось распознать изображение, попробуйте другое фото',
  NOT_FOUND: 'Карточка не найдена',
  UNKNOWN: 'Что-то пошло не так'
}

export const imageMessages = {
  notImage: 'Выберите изображение (JPG, PNG, HEIC)',
  tooLarge: 'Файл слишком большой',
  cannotDecode: 'Не удалось прочитать фото, попробуйте ещё раз'
} as const

export const uiMessages = {
  linkCopied: 'Ссылка скопирована',
  shareFailed: 'Не удалось поделиться ссылкой',
  wineNotFound: 'Карточка вина не найдена',
  suggestSent: 'Спасибо, передадим редакции каталога',
  suggestFailed: 'Не удалось отправить предложение, попробуйте позже'
} as const
