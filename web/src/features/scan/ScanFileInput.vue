<script setup lang="ts">
import { ref } from 'vue'
import type { ScanSource } from '@/services/types/api-types'

const emit = defineEmits<{ select: [file: File, source: ScanSource] }>()

const cameraInput = ref<HTMLInputElement | null>(null)
const galleryInput = ref<HTMLInputElement | null>(null)

const readFile = (event: Event, source: ScanSource): void => {
  const input = event.target
  if (!(input instanceof HTMLInputElement)) {
    return
  }
  const file = input.files?.[0]
  input.value = ''
  if (file === undefined) {
    return
  }
  emit('select', file, source)
}

const onCameraChange = (event: Event): void => readFile(event, 'camera')
const onGalleryChange = (event: Event): void => readFile(event, 'gallery')

const openCamera = (): void => cameraInput.value?.click()
const openGallery = (): void => galleryInput.value?.click()

defineExpose({ openCamera, openGallery })
</script>

<template>
  <input
    ref="cameraInput"
    class="visually-hidden"
    type="file"
    accept="image/*"
    capture="environment"
    tabindex="-1"
    aria-hidden="true"
    @change="onCameraChange"
  >
  <input
    ref="galleryInput"
    class="visually-hidden"
    type="file"
    accept="image/*"
    tabindex="-1"
    aria-hidden="true"
    @change="onGalleryChange"
  >
</template>
