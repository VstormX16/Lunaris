import { createContext, useCallback, useContext, useMemo, useState, type ReactNode } from 'react'

/** simple: plain words for everyone. advanced: the full technical picture. */
export type Level = 'simple' | 'advanced'
const KEY = 'lunaris-level'

const read = (): Level => {
  try {
    return localStorage.getItem(KEY) === 'simple' ? 'simple' : 'advanced'
  } catch {
    return 'advanced'
  }
}

const Ctx = createContext<{ level: Level; setLevel: (l: Level) => void }>({ level: 'advanced', setLevel: () => undefined })

export function SettingsProvider({ children }: { children: ReactNode }) {
  const [level, setState] = useState<Level>(read)
  const setLevel = useCallback((l: Level) => {
    setState(l)
    try {
      localStorage.setItem(KEY, l)
    } catch {
      /* not remembered */
    }
  }, [])
  const value = useMemo(() => ({ level, setLevel }), [level, setLevel])
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>
}

export const useSettings = () => useContext(Ctx)
