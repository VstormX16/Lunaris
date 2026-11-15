import { Plus } from 'lucide-react'
import type { AssistantAction } from '../api'
import { Body } from './AssistantPanel'
import ErrorBoundary from './ErrorBoundary'
import type { Chat } from '../lib/chat'
import { useSettings } from '../lib/settings'
import { useAssistant } from '../lib/useAssistant'

type Free = { date?: string; vehicleId?: string; lat?: number; lon?: number; site?: string; site_given?: boolean }

type Props = {
  initial: Chat
  onChange: (c: Chat) => void
  onNew: () => void
}

function Chatting({ initial, onChange, onNew }: Props) {
  const { level } = useSettings()
  const a = useAssistant({ scenarios: () => [], free: true, initial, onChange, onApply: (x) => apply(x), level })
  // "Apply that date" in the free chat: there is no page to change, so the conversation itself takes the value and uses it from now on
  function apply(x: AssistantAction) {
    a.patchMemory((m) => {
      const f = { ...((m.free as Free | undefined) ?? {}) }
      if (x.date) f.date = x.date
      if (x.vehicleId) f.vehicleId = x.vehicleId
      if (x.lat !== undefined && x.lon !== undefined) {
        f.lat = x.lat; f.lon = x.lon; f.site_given = true
        f.site = `${x.lat.toFixed(2)}, ${(x.lon > 180 ? x.lon - 360 : x.lon).toFixed(2)}`
      }
      return { ...m, free: f, suggestion: null }
    })
  }
  return (
    <main className="flex min-w-0 flex-1 flex-col">
      <div className="mx-auto flex h-full w-full max-w-[860px] flex-col px-10 pb-8 pt-10">
        <div className="flex items-start justify-between gap-6 border-b border-line pb-4">
          <div>
            <h1 className="text-[32px] font-medium text-brand">AI Assistant</h1>
            <p className="mt-1 text-[15px] text-muted">Ask in your own words. Every number is computed from real NASA data. Conversations are kept in History.</p>
          </div>
          {a.chat.msgs.length > 0 && (
            <button onClick={onNew} className="flex shrink-0 items-center gap-1.5 pt-2 text-[15px] text-brand hover:underline hover:underline-offset-4"><Plus size={16} /> New chat</button>
          )}
        </div>
        <section translate="no" className="notranslate flex min-h-0 flex-1 flex-col" aria-label="Assistant chat">
          <Body chat={a.chat} busy={a.busy} onSend={(q) => void a.send(q)} onApply={apply} two={false} free />
        </section>
      </div>
    </main>
  )
}

export default function AssistantPage(props: Props) {
  return (
    <ErrorBoundary fallback={(e, reset) => (
      <main className="flex flex-1 flex-col items-center justify-center gap-3 text-center">
        <p className="text-ink">The assistant had a problem.</p>
        <p className="max-w-[480px] break-words text-[12px] text-muted">{e.message.slice(0, 200)}</p>
        <button onClick={reset} className="text-brand hover:underline">Try again</button>
      </main>
    )}>
      <Chatting {...props} />
    </ErrorBoundary>
  )
}
