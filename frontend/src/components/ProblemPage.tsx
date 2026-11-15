import { RotateCcw } from 'lucide-react'

type Props = {
  /** 404: the address does not exist. error: a screen stopped working. */
  kind: '404' | 'error'
  detail?: string
  onHome: () => void
  onRetry?: () => void
}

const COPY = {
  '404': { code: '404', title: 'This page is not on the map', text: 'The address you opened does not exist in Lunaris. Nothing is broken; you may have followed an old link or mistyped it.' },
  error: { code: 'Oops', title: 'This screen ran into a problem', text: 'Something went wrong while showing this screen. Your saved operations and conversations are safe; you can try again or start from the beginning.' },
} as const

/** One page for every kind of "something is wrong": a missing address, or a screen that crashed. */
export default function ProblemPage({ kind, detail, onHome, onRetry }: Props) {
  const c = COPY[kind]
  return (
    <main className="flex flex-1 items-center justify-center px-10">
      <div className="step-in max-w-[560px]">
        <div className="text-[96px] font-medium leading-none tracking-tight text-brand/80 tabular-nums">{c.code}</div>
        <h1 className="mt-4 text-[30px] font-medium text-strong">{c.title}</h1>
        <p className="mt-2 text-[17px] leading-relaxed text-muted">{c.text}</p>
        {detail && <p className="mt-3 break-words border-l-2 border-line pl-3 text-[13px] text-muted">{detail.slice(0, 240)}</p>}
        <div className="mt-8 flex items-center gap-8 border-t border-line pt-5 text-[17px]">
          <button onClick={onHome} className="text-brand hover:underline hover:underline-offset-4">Go to the start</button>
          {onRetry && (
            <button onClick={onRetry} className="flex items-center gap-2 text-muted hover:text-brand"><RotateCcw size={16} /> Try again</button>
          )}
        </div>
      </div>
    </main>
  )
}
