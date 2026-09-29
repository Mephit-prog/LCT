import { onBeforeUnmount, watch, type Ref } from 'vue'

const focusableSelector = 'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])'

const getFocusable = (root: HTMLElement): HTMLElement[] => {
  return Array.from(root.querySelectorAll<HTMLElement>(focusableSelector))
}

export const useFocusTrap = (container: Ref<HTMLElement | null>, isActive: Ref<boolean>): void => {
  let previouslyFocused: HTMLElement | null = null

  const onKeydown = (event: KeyboardEvent): void => {
    if (event.key !== 'Tab' || container.value === null) {
      return
    }
    const focusable = getFocusable(container.value)
    const first = focusable[0]
    const last = focusable[focusable.length - 1]
    if (first === undefined || last === undefined) {
      event.preventDefault()
      return
    }
    const active = document.activeElement
    if (event.shiftKey && (active === first || !container.value.contains(active))) {
      event.preventDefault()
      last.focus()
      return
    }
    if (!event.shiftKey && active === last) {
      event.preventDefault()
      first.focus()
    }
  }

  const activate = (): void => {
    previouslyFocused = document.activeElement instanceof HTMLElement ? document.activeElement : null
    document.addEventListener('keydown', onKeydown)
    setTimeout(() => {
      if (container.value === null || !isActive.value) {
        return
      }
      const target = getFocusable(container.value)[0] ?? container.value
      target.focus()
    }, 0)
  }

  const deactivate = (): void => {
    document.removeEventListener('keydown', onKeydown)
    previouslyFocused?.focus()
    previouslyFocused = null
  }

  watch(isActive, (value, previous) => {
    if (value && !previous) {
      activate()
      return
    }
    if (!value && previous) {
      deactivate()
    }
  }, { immediate: true })

  onBeforeUnmount(() => {
    if (isActive.value) {
      deactivate()
    }
  })
}
