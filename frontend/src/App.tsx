import { useCallback, useEffect, useRef, useState } from 'react'
import { createSession, deleteSession, getSession, getVehicles, listSessions, updateSession, vehicleNames, type AssistantAction, type SessionSummary } from './api'
import AssistantPage from './components/AssistantPage'
import AnalysisPage, { type Open360 } from './components/AnalysisPage'
import ErrorBoundary from './components/ErrorBoundary'
import HomeScreen from './components/HomeScreen'
import ProblemPage from './components/ProblemPage'
import Moon360Page from './components/Moon360Page'
import MoonMapPage from './components/MoonMapPage'
import LandingMapPage from './components/LandingMapPage'
import PlannerPage from './components/PlannerPage'
import MoonTimelinePage from './components/MoonTimelinePage'
import Sidebar, { type View } from './components/Sidebar'
import SpacecraftsPage from './components/SpacecraftsPage'
import SpaceTimePage from './components/SpaceTimePage'
import Wizard from './components/wizard/Wizard'
import { spacecrafts } from './data/spacecrafts'
import { emptyChat, fromStored, toStored, type Chat } from './lib/chat'
import { formatDate } from './lib/input'
import { initialWizard, scenarioB, type Scenario, type WizardState } from './types'

const WIDE_VIEWS: View[] = ['wizard', 'analysis', 'planner', 'landmap', 'moonmap', 'timeline', 'view360']

const NAMES = Object.fromEntries(spacecrafts.map((c) => [c.id, c.name]))
const describe = (s: Scenario) => `${NAMES[s.vehicleId] ?? vehicleNames[s.vehicleId] ?? s.vehicleId} · ${formatDate(s.date)} · ${s.lat}, ${s.lon}`
/** The History entry's title: what the operation is, in one line. */
const titleOf = (w: WizardState) => {
  const b = scenarioB(w)
  return b ? `${describe(w.a)}  ⇄  ${describe(b)}` : describe(w.a)
}

export default function App() {
  // there is one real address; anything else typed or followed is a page that does not exist
  const [notFound, setNotFound] = useState(() => !['/', '/index.html'].includes(window.location.pathname))
  const [collapsed, setCollapsed] = useState(() => typeof window !== 'undefined' && window.matchMedia('(max-width: 1099px)').matches)
  const [view, setView] = useState<View>('home')
  // the sidebar folds to its icons where a page needs the width (the operation flow, the maps, 360) and on narrow
  // windows; it opens again on the other pages. The toggle still works in between.
  useEffect(() => {
    const mq = window.matchMedia('(max-width: 1099px)')
    const fold = () => setCollapsed(mq.matches || WIDE_VIEWS.includes(view))
    fold()
    mq.addEventListener('change', fold)
    return () => mq.removeEventListener('change', fold)
  }, [view])
  const [wizard, setWizard] = useState<WizardState>(initialWizard)
  const [sessions, setSessions] = useState<SessionSummary[]>([])
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [chat, setChat] = useState<Chat>(emptyChat) // the conversation the analysis page opens with
  const [pageKey, setPageKey] = useState(0) // a new key makes the analysis page start over (another operation was opened)
  const [target360, setTarget360] = useState<Open360 | null>(null)
  // the free AI chat: one saved conversation in History, created with its first message
  const [freeChat, setFreeChat] = useState<Chat>(emptyChat)
  const [freeId, setFreeId] = useState<string | null>(null)
  const [freeKey, setFreeKey] = useState(0)
  const freeIdRef = useRef<string | null>(null)
  const freeLatest = useRef<Chat>(emptyChat)
  const freeTimer = useRef<ReturnType<typeof setTimeout>>(undefined)
  const freeCreating = useRef(false)
  const idRef = useRef<string | null>(null)
  idRef.current = sessionId
  const pendingChat = useRef<Chat | null>(null)
  const chatTimer = useRef<ReturnType<typeof setTimeout>>(undefined)

  const refresh = useCallback(() => {
    listSessions().then(setSessions).catch(() => undefined)
  }, [])

  // warm the vehicle list once so the first step 2 render is instant when the server is up; load the History list
  useEffect(() => {
    getVehicles().catch(() => undefined)
    refresh()
  }, [refresh])

  const goHome = () => {
    window.history.replaceState(null, '', '/')
    setNotFound(false)
    setView('home')
  }

  const start = () => {
    setWizard(initialWizard) // "Start New Operation" always begins a fresh operation
    setSessionId(null)
    setChat(emptyChat)
    pendingChat.current = null
    setPageKey((k) => k + 1)
    setView('wizard')
  }

  const navigate = (v: View) => {
    if (notFound) goHome()
    if (v === 'wizard') return start()
    if (v === 'view360') setTarget360(null) // opened from the menu: pick a place yourself
    setView(v)
  }

  // ---- saving: the operation (its inputs) when Analyze is pressed, its conversation as it grows ----
  const saveChat = useCallback((c: Chat) => {
    pendingChat.current = c
    setChat(c) // so the page, if it is left and opened again, comes back with the conversation
    clearTimeout(chatTimer.current)
    chatTimer.current = setTimeout(() => {
      const id = idRef.current
      if (id && pendingChat.current) updateSession(id, { chat: toStored(pendingChat.current) }).catch(() => undefined)
    }, 600)
  }, [])

  /** Opens the analysis of one scenario as a new saved operation (the Mission Planner's "analyze this date"). */
  const analyzeScenario = (a: Scenario) => {
    const w: WizardState = { ...initialWizard, a, mode: 'instant', step: 4, editing: 'a' }
    setWizard(w)
    setSessionId(null)
    idRef.current = null
    setChat(emptyChat)
    pendingChat.current = null
    setPageKey((k) => k + 1)
    setView('analysis')
    createSession(titleOf(w), w, toStored(emptyChat)).then((x) => { setSessionId(x.id); refresh() }).catch(() => undefined)
  }

  const analyze = () => {
    const w = { ...wizard, editing: 'a' as const }
    setWizard(w)
    setView('analysis')
    const body = { name: titleOf(w), state: w }
    const done = (id: string) => {
      setSessionId(id)
      refresh()
      if (pendingChat.current && pendingChat.current.msgs.length) updateSession(id, { chat: toStored(pendingChat.current) }).catch(() => undefined)
    }
    if (idRef.current) updateSession(idRef.current, body).then((s) => done(s.id)).catch(() => undefined)
    else createSession(body.name, body.state, toStored(pendingChat.current ?? emptyChat)).then((s) => done(s.id)).catch(() => undefined)
  }

  // when the assistant (or anything else) changes the inputs of a saved operation, keep the saved copy in step
  const lastSaved = useRef('')
  useEffect(() => {
    if (view !== 'analysis' || !sessionId) return
    const body = JSON.stringify({ a: wizard.a, b: wizard.b, mode: wizard.mode, kind: wizard.compareKind })
    if (body === lastSaved.current) return
    const first = lastSaved.current === ''
    lastSaved.current = body
    if (first) return // just opened or just created: nothing changed yet
    const t = setTimeout(() => updateSession(sessionId, { name: titleOf(wizard), state: { ...wizard, editing: 'a' } }).then(refresh).catch(() => undefined), 500)
    return () => clearTimeout(t)
  }, [wizard, view, sessionId, refresh])
  useEffect(() => { lastSaved.current = '' }, [sessionId])

  const newFreeChat = () => {
    clearTimeout(freeTimer.current)
    freeIdRef.current = null
    freeLatest.current = emptyChat
    setFreeId(null)
    setFreeChat(emptyChat)
    setFreeKey((k) => k + 1)
  }

  const saveFreeChat = useCallback((c: Chat) => {
    freeLatest.current = c
    setFreeChat(c) // switching tabs and coming back keeps the conversation
    if (!c.msgs.length) return
    clearTimeout(freeTimer.current)
    freeTimer.current = setTimeout(() => {
      const chatNow = toStored(freeLatest.current)
      if (freeIdRef.current) {
        updateSession(freeIdRef.current, { chat: chatNow }).catch(() => undefined)
      } else if (!freeCreating.current) {
        freeCreating.current = true
        const first = freeLatest.current.msgs.find((m) => m.from === 'user')
        const title = first && first.from === 'user' ? first.text.replace(/\s+/g, ' ').slice(0, 60) : 'Conversation'
        createSession(title, { kind: 'chat' }, chatNow)
          .then((s) => {
            freeIdRef.current = s.id
            setFreeId(s.id)
            refresh()
            updateSession(s.id, { chat: toStored(freeLatest.current) }).catch(() => undefined) // anything said while it was being created
          })
          .catch(() => undefined)
          .finally(() => { freeCreating.current = false })
      }
    }, 500)
  }, [refresh])

  const openSession = async (id: string) => {
    try {
      const s = await getSession(id)
      if ((s.state as { kind?: string } | null)?.kind === 'chat') {
        const c = fromStored(s.chat)
        clearTimeout(freeTimer.current)
        freeIdRef.current = id
        freeLatest.current = c
        setFreeId(id)
        setFreeChat(c)
        setFreeKey((k) => k + 1)
        setView('assistant')
        return
      }
      const w = { ...initialWizard, ...(s.state as WizardState), step: 4 as const, editing: 'a' as const }
      setWizard(w)
      const c = fromStored(s.chat)
      setChat(c)
      pendingChat.current = c
      setSessionId(id)
      setPageKey((k) => k + 1)
      setView('analysis')
    } catch {
      refresh() // it was deleted elsewhere
    }
  }

  const removeSession = async (id: string) => {
    await deleteSession(id).catch(() => undefined)
    if (id === freeId) newFreeChat()
    if (id === sessionId) {
      setSessionId(null)
      setChat(emptyChat)
      pendingChat.current = null
      setWizard(initialWizard)
      setView('home')
    }
    refresh()
  }

  // the assistant can change the date, vehicle or place of A or B; the analysis page reruns when the answers change
  const apply = (a: AssistantAction) =>
    setWizard((w) => {
      const change: Partial<Scenario> = {
        ...(a.date ? { date: a.date } : {}),
        ...(a.vehicleId ? { vehicleId: a.vehicleId } : {}),
        ...(a.lat !== undefined && a.lon !== undefined ? { lat: String(a.lat), lon: String(a.lon > 180 ? a.lon - 360 : a.lon) } : {}),
      }
      if (a.scenario === 'b') {
        // B becomes its own full scenario, so the change cannot leak into A
        const current = scenarioB(w) ?? w.a
        return { ...w, mode: 'full', compareKind: null, b: { ...current, ...change } }
      }
      return { ...w, a: { ...w.a, ...change } }
    })

  const open360 = (t: Open360) => {
    setTarget360(t)
    setView('view360')
  }

  return (
    <div className="flex h-full">
      <Sidebar
        collapsed={collapsed}
        onToggle={() => setCollapsed((c) => !c)}
        view={view}
        onNavigate={navigate}
        sessions={sessions}
        activeSession={view === 'assistant' ? freeId : sessionId}
        onOpenSession={(id) => void openSession(id)}
        onDeleteSession={(id) => void removeSession(id)}
      />
      <ErrorBoundary key={view} fallback={(e, reset) => <ProblemPage kind="error" detail={e.message} onRetry={reset} onHome={() => setView('home')} />}>
      {notFound && <ProblemPage kind="404" onHome={goHome} />}
      {!notFound && view === 'spacecrafts' && <SpacecraftsPage />}
      {!notFound && view === 'spacetime' && <SpaceTimePage />}
      {!notFound && view === 'moonmap' && <MoonMapPage />}
      {!notFound && view === 'landmap' && <LandingMapPage />}
      {!notFound && view === 'planner' && <PlannerPage onAnalyze={analyzeScenario} />}
      {!notFound && view === 'timeline' && <MoonTimelinePage />}
      {!notFound && view === 'assistant' && <AssistantPage key={freeKey} initial={freeChat} onChange={saveFreeChat} onNew={newFreeChat} />}
      {!notFound && view === 'view360' && <Moon360Page key={JSON.stringify(target360)} target={target360} onBack={target360 ? () => setView('analysis') : undefined} />}
      {!notFound && view === 'home' && <HomeScreen onStart={start} sessions={sessions} onOpen={(id) => void openSession(id)} onNavigate={navigate} />}
      {!notFound && view === 'wizard' && <Wizard state={wizard} setState={setWizard} onAnalyze={analyze} />}
      {!notFound && view === 'analysis' && (
        <AnalysisPage
          key={pageKey}
          state={wizard}
          vehicleNames={{ ...vehicleNames, ...NAMES }}
          initialChat={chat}
          onChat={saveChat}
          onApply={apply}
          onOpen360={open360}
          onEdit={() => {
            setWizard((w) => ({ ...w, step: 4, editing: 'a' }))
            setView('wizard')
          }}
        />
      )}
      </ErrorBoundary>
    </div>
  )
}
