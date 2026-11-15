// Grain on interactive surfaces (elements with the `grain` class): the grain gathers around the pointer while
// hovering, and a press sends a ring of grain outward like a wave. Styles are in index.css (.grain, .grain-ripple).

const reduced = () => window.matchMedia('(prefers-reduced-motion: reduce)').matches

function grainOf(target: EventTarget | null): HTMLElement | null {
  return target instanceof Element ? (target.closest('.grain') as HTMLElement | null) : null
}

export function installGrain() {
  document.addEventListener(
    'pointermove',
    (e) => {
      const el = grainOf(e.target)
      if (!el) return
      const r = el.getBoundingClientRect()
      el.style.setProperty('--gx', `${e.clientX - r.left}px`)
      el.style.setProperty('--gy', `${e.clientY - r.top}px`)
    },
    { passive: true },
  )
  document.addEventListener('pointerdown', (e) => {
    const el = grainOf(e.target)
    if (!el || reduced() || (el as HTMLButtonElement).disabled) return
    const r = el.getBoundingClientRect()
    const d = Math.hypot(r.width, r.height) * 2
    const wave = document.createElement('span')
    wave.className = 'grain-ripple'
    wave.setAttribute('aria-hidden', 'true')
    wave.style.width = wave.style.height = `${d}px`
    wave.style.left = `${e.clientX - r.left - d / 2}px`
    wave.style.top = `${e.clientY - r.top - d / 2}px`
    el.appendChild(wave)
    wave.addEventListener('animationend', () => wave.remove(), { once: true })
  })
}
