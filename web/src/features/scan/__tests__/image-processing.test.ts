import { describe, expect, it } from 'vitest'
import {
  computeTargetSize,
  createImagePrepareError,
  isImagePrepareError,
  planNextAttempt,
  prepareImage,
  validateImageFile
} from '../image-processing'

const makeFile = (name: string, type: string, size: number): File => {
  const file = new File([new Uint8Array(1)], name, { type })
  Object.defineProperty(file, 'size', { value: size })
  return file
}

describe('validateImageFile', () => {
  it('отклоняет файлы, которые не являются изображениями', () => {
    expect(validateImageFile(makeFile('doc.pdf', 'application/pdf', 1000), 25_000_000)).toBe('not_image')
  })

  it('принимает HEIC без mime-типа по расширению', () => {
    expect(validateImageFile(makeFile('IMG_0001.HEIC', '', 1000), 25_000_000)).toBeNull()
  })

  it('отклоняет слишком большие файлы', () => {
    expect(validateImageFile(makeFile('big.jpg', 'image/jpeg', 26_000_000), 25_000_000)).toBe('too_large')
  })

  it('принимает корректное изображение', () => {
    expect(validateImageFile(makeFile('label.jpg', 'image/jpeg', 3_000_000), 25_000_000)).toBeNull()
  })
})

describe('computeTargetSize', () => {
  it('уменьшает по длинной стороне с сохранением пропорций', () => {
    expect(computeTargetSize({ width: 4000, height: 3000 }, 1280)).toEqual({ width: 1280, height: 960 })
    expect(computeTargetSize({ width: 3000, height: 4000 }, 1280)).toEqual({ width: 960, height: 1280 })
  })

  it('не увеличивает маленькие изображения', () => {
    expect(computeTargetSize({ width: 800, height: 600 }, 1280)).toEqual({ width: 800, height: 600 })
  })
})

describe('planNextAttempt', () => {
  it('останавливается, когда размер укладывается в цель', () => {
    expect(planNextAttempt({ maxSide: 1280, quality: 0.85 }, 400_000, 512_000)).toBeNull()
  })

  it('понижает качество шагом 0.1', () => {
    expect(planNextAttempt({ maxSide: 1280, quality: 0.85 }, 900_000, 512_000)).toEqual({ maxSide: 1280, quality: 0.75 })
  })

  it('после исчерпания качества уменьшает сторону', () => {
    expect(planNextAttempt({ maxSide: 1280, quality: 0.5 }, 900_000, 512_000)).toEqual({ maxSide: 960, quality: 0.7 })
  })

  it('прекращает попытки ниже минимальной стороны', () => {
    expect(planNextAttempt({ maxSide: 640, quality: 0.5 }, 900_000, 512_000)).toBeNull()
  })
})

describe('prepareImage', () => {
  it('бросает типизированную ошибку валидации до декодирования', async () => {
    await expect(prepareImage(makeFile('doc.pdf', 'application/pdf', 100), {
      maxSide: 1280,
      quality: 0.85,
      targetBytes: 512_000,
      maxInputBytes: 25_000_000
    })).rejects.toMatchObject({ name: 'ImagePrepareError', code: 'not_image' })
  })

  it('возвращает ошибку декодирования для битого изображения', async () => {
    const file = new File([new Uint8Array([1, 2, 3])], 'broken.jpg', { type: 'image/jpeg' })
    await expect(prepareImage(file, {
      maxSide: 1280,
      quality: 0.85,
      targetBytes: 512_000,
      maxInputBytes: 25_000_000
    })).rejects.toSatisfy((error: unknown) => isImagePrepareError(error) && error.code === 'decode_failed')
  })
})

describe('createImagePrepareError', () => {
  it('содержит русское сообщение для пользователя', () => {
    expect(createImagePrepareError('too_large').message).toBe('Файл слишком большой')
  })
})
