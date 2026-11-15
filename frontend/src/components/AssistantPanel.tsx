import { PanelBottom, PictureInPicture2 } from 'lucide-react'
import { useEffect, useRef, useState, type ReactNode } from 'react'
import type { AssistantAction, AssistantBlock } from '../api'
import type { Chat } from '../lib/chat'
import ErrorBoundary from './ErrorBoundary'
import FloatingFrame from './FloatingFrame'

const welcome = (two: boolean, free = false): AssistantBlock[] => free ? [
  { type: 'text', text: 'Ask me anything about landing on the Moon: pick a date, a vehicle or a place in your own words, for example which vehicle to send on a given day.' },
  { type: 'text', text: 'Doğrudan Türkçe ya da İngilizce sorabilirsin.' },
  { type: 'chips', items: ['Which vehicle should we send on September 12?', 'How does Shackleton look on March 3?', 'What can you do?'] },
] : [
  { type: 'text', text: two ? 'Ask me about either result, or about the two together: which is better, why one is lower, what would lift it.' : 'Ask me about this analysis: why the score is what it is, how another date or vehicle would do, what a term means.' },
  { type: 'text', text: 'Bu analiz hakkında bana Türkçe ya da İngilizce soru sorabilirsin.' },
  { type: 'chips', items: two ? ['Compare the two results', 'Summarize scenario A', 'Summarize scenario B'] : ['Summarize the result', 'Why is the score low?', 'Tarihi nasıl oynatabiliriz?'] },
]
const WORKING = ['Reading your question…', 'Running the engine on the real Sun and Earth positions…', 'Writing the answer…']

/** **bold** markers from the assistant, nothing else. */
function Rich({ text }: { text: string }) {
  return (
    <>
      {text.split(/(\*\*[^*]+\*\*)/g).map((p, i) => (p.startsWith('**') ? <strong key={i} className="font-semibold text-strong">{p.slice(2, -2)}</strong> : <span key={i}>{p}</span>))}
    </>
  )
}

const str = (v: unknown) => (typeof v === 'string' ? v : typeof v === 'number' ? String(v) : '')

function Block({ b, onAsk, onApply }: { b: AssistantBlock; onAsk: (q: string) => void; onApply: (a: AssistantAction) => void }) {
  if (b.type === 'text') return <p className="whitespace-pre-line text-[15px] leading-relaxed text-ink"><Rich text={str(b.text)} /></p>
  if (b.type === 'bullets')
    return (
      <ul className="list-disc space-y-1 pl-5 text-[15px] leading-snug text-ink marker:text-brand/60">
        {(b.items ?? []).map((i, k) => <li key={k}><Rich text={str(i)} /></li>)}
      </ul>
    )
  if (b.type === 'table')
    return (
      <div className="overflow-x-auto">
        <div className="mb-1 text-[12px] uppercase tracking-wide text-muted">{str(b.title)}</div>
        <table className="w-full border-collapse text-left text-[14px] tabular-nums">
          <thead>
            <tr>{(b.columns ?? []).map((c, k) => <th key={k} className="pb-1.5 pr-4 text-[12px] font-normal uppercase tracking-wide text-muted">{str(c)}</th>)}</tr>
          </thead>
          <tbody>
            {(b.rows ?? []).map((r, i) => (
              <tr key={i} className="border-t border-line">
                {(r ?? []).map((c, j) => <td key={j} className={`py-1.5 pr-4 align-top ${j === 0 ? 'text-ink' : 'text-strong'}`}>{str(c)}</td>)}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    )
  if (b.type === 'chips')
    return (
      <div className="flex flex-wrap gap-2">
        {(b.items ?? []).map((c) => (
          <button key={c} onClick={() => onAsk(c)} className="rounded-full border border-line bg-white/70 px-3 py-1 text-[13px] text-ink transition hover:border-brand hover:text-brand">{c}</button>
        ))}
      </div>
    )
  if (b.type !== 'actions') return null // a block type this screen does not know: skip it rather than crash
  return (
    <div className="flex flex-wrap gap-2">
      {(b.buttons ?? []).map((x) => (
        <button key={x.label} onClick={() => onApply(x.action)} className="rounded-[8px] border border-brand bg-white px-4 py-1.5 text-[14px] text-ink transition hover:text-brand">{x.label}</button>
      ))}
    </div>
  )
}

export type ScenarioLine = { label: 'A' | 'B'; text: string }

type Props = {
  lines: ScenarioLine[]
  chat: Chat
  busy: boolean
  onSend: (q: string) => void
  onApply: (a: AssistantAction) => void
  floating: boolean
  onToggleFloat: () => void
}

export function Body({ chat, busy, onSend, onApply, two, free = false }: { chat: Chat; busy: boolean; onSend: (q: string) => void; onApply: (a: AssistantAction) => void; two: boolean; free?: boolean }) {
  const [text, setText] = useState('')
  const [step, setStep] = useState(0)
  const end = useRef<HTMLDivElement>(null)

  useEffect(() => {
    end.current?.scrollIntoView({ behavior: 'smooth', block: 'end' }) // not returned: React would take a return value for the cleanup function
  }, [chat.msgs.length, busy])
  useEffect(() => {
    if (!busy) return
    setStep(0)
    const t = setInterval(() => setStep((s) => Math.min(s + 1, WORKING.length - 1)), 1100)
    return () => clearInterval(t)
  }, [busy])

  const submit = () => {
    const q = text.trim()
    if (!q) return
    setText('')
    onSend(q)
  }

  return (
    <>
      <div className="flex-1 space-y-5 overflow-y-auto px-5 py-5">
        <div className="space-y-3">{welcome(two, free).map((b, j) => <Block key={j} b={b} onAsk={onSend} onApply={onApply} />)}</div>
        {chat.msgs.map((m, i) =>
          m.from === 'user' ? (
            <div key={i} className="flex justify-end">
              <div className="reveal max-w-[85%] rounded-[12px] bg-active px-3.5 py-2 text-[15px] text-ink">{m.text}</div>
            </div>
          ) : (
            <div key={i} className="space-y-3">
              {m.blocks.map((b, j) => <div key={j} className="reveal" style={{ '--i': j } as React.CSSProperties}><Block b={b} onAsk={onSend} onApply={onApply} /></div>)}
            </div>
          ),
        )}
        {busy && <p key={step} className="gem-text reveal text-[14px]" role="status">{WORKING[step]}</p>}
        <div ref={end} />
      </div>
      <form onSubmit={(e) => { e.preventDefault(); submit() }} className="border-t border-line px-5 py-3">
        <div className="flex items-center gap-3 border-b border-line pb-1.5 transition-colors focus-within:border-brand">
          <input
            id="assistant-input"
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Ask a question…  /  Bir soru sor…"
            aria-label="Your question"
            maxLength={500}
            className="min-w-0 flex-1 bg-transparent py-1 text-[16px] text-ink outline-none placeholder:text-muted/60"
          />
          <button type="submit" disabled={!text.trim()} className="text-[15px] text-brand transition disabled:text-muted/50">Send</button>
        </div>
      </form>
    </>
  )
}

function Title({ lines, floating, onToggleFloat }: { lines: ScenarioLine[]; floating: boolean; onToggleFloat: () => void }) {
  return (
    <div className="flex items-start justify-between gap-3">
      <div className="min-w-0">
        <div className="text-[20px] font-medium text-brand">Assistant</div>
        {lines.map((l) => (
          <div key={l.label} className="mt-0.5 truncate text-[13px] text-muted" title={l.text}>
            {lines.length > 1 && <span className="mr-1.5 text-brand">{l.label}</span>}{l.text}
          </div>
        ))}
      </div>
      <button onClick={onToggleFloat} className="flex shrink-0 items-center gap-1.5 rounded-[8px] px-2.5 py-1.5 text-[14px] text-muted transition hover:bg-active hover:text-brand" aria-label={floating ? 'Dock the assistant into the page' : 'Float the assistant as a window'}>
        {floating ? <PanelBottom size={16} /> : <PictureInPicture2 size={16} />} {floating ? 'Dock' : 'Float'}
      </button>
    </div>
  )
}

function Panel({ lines, chat, busy, onSend, onApply, floating, onToggleFloat }: Props) {
  const body = <Body chat={chat} busy={busy} onSend={onSend} onApply={onApply} two={lines.length > 1} />
  if (floating) {
    return (
      <FloatingFrame header={(drag) => <div {...drag} className={`border-b border-line px-5 py-3 ${drag.className}`}><Title lines={lines} floating onToggleFloat={onToggleFloat} /></div>}>
        {body}
      </FloatingFrame>
    )
  }
  // docked: an ordinary section of the page
  return (
    <section translate="no" className="notranslate mt-10 flex h-[600px] flex-col overflow-hidden rounded-[12px] border border-line bg-white/60" aria-label="Assistant">
      <div className="border-b border-line px-5 py-3"><Title lines={lines} floating={false} onToggleFloat={onToggleFloat} /></div>
      {body}
    </section>
  )
}

/** The assistant can fail (bad answer, lost connection, a browser extension touching the page) without taking the analysis down with it. */
export default function AssistantPanel(props: Props) {
  const fallback = (error: Error, reset: () => void): ReactNode => (
    <section translate="no" className="notranslate mt-10 flex flex-col gap-3 rounded-[12px] border border-line p-6" aria-label="Assistant">
      <div className="text-[20px] font-medium text-brand">Assistant</div>
      <p className="text-[15px] text-ink">The assistant had a problem and was stopped; your analysis is untouched.</p>
      <p className="break-words text-[12px] text-muted">{error.message.slice(0, 200)}</p>
      <button onClick={reset} className="self-start text-[15px] text-brand hover:underline">Try again</button>
    </section>
  )
  return (
    <ErrorBoundary fallback={fallback}>
      <Panel {...props} />
    </ErrorBoundary>
  )
}
