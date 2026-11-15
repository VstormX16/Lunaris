import { useCallback, useEffect, useRef, useState } from 'react'
import { askAssistant, type AssistantAction, type AssistantScenario } from '../api'
import type { Chat } from './chat'
import type { Level } from './settings'

type Options = {
  /** the scenario(s) on the page right now: A, and B when two results are shown */
  scenarios: () => AssistantScenario[]
  initial: Chat
  onChange?: (c: Chat) => void
  onApply: (a: AssistantAction) => void
  /** free chat: no analysis on screen, the assistant builds the scenario from what is said */
  free?: boolean
  /** how technical the answers should be */
  level?: Level
}

/**
 * The conversation with the assistant. It lives above the panel so the chat survives docking and floating the panel,
 * and a message sent while an answer is still being worked out is queued instead of being dropped.
 */
export function useAssistant({ scenarios, initial, onChange, onApply, free = false, level = 'advanced' }: Options) {
  const [chat, setChat] = useState<Chat>(initial)
  const [busy, setBusy] = useState(false)
  const chatRef = useRef(chat)
  const queue = useRef<string[]>([])
  const running = useRef(false)
  const opts = useRef({ scenarios, onChange, onApply, level })
  opts.current = { scenarios, onChange, onApply, level }

  const update = useCallback((fn: (c: Chat) => Chat) => {
    const next = fn(chatRef.current)
    chatRef.current = next
    setChat(next)
    opts.current.onChange?.(next)
  }, [])

  const send = useCallback(async (text: string) => {
    const message = text.trim()
    if (!message) return
    update((c) => ({ ...c, msgs: [...c.msgs, { from: 'user', text: message }] }))
    queue.current.push(message)
    if (running.current) return
    running.current = true
    setBusy(true)
    while (queue.current.length) {
      const m = queue.current.shift()!
      try {
        const scs = opts.current.scenarios()
        if (!scs.length && !free) throw new Error('Run an analysis first, then I can talk about it.')
        const r = await askAssistant(m, scs, chatRef.current.memory, free, opts.current.level)
        update((c) => ({ msgs: [...c.msgs, { from: 'bot', blocks: r.reply.blocks }], memory: r.memory }))
        r.reply.actions?.forEach(opts.current.onApply)
      } catch (e) {
        update((c) => ({ ...c, msgs: [...c.msgs, { from: 'bot', blocks: [{ type: 'text', text: (e as Error).message }] }] }))
      }
    }
    running.current = false
    setBusy(false)
  }, [update])

  // the first question, when the page opens it (kept out of render so it is asked once)
  const asked = useRef(false)
  const askOnce = useCallback((q: string) => {
    if (asked.current) return
    asked.current = true
    void send(q)
  }, [send])
  useEffect(() => () => { queue.current = [] }, [])

  /** change the assistant's memory from outside (the free chat takes an applied date or place), optionally with a note in the conversation */
  const patchMemory = useCallback((fn: (m: Record<string, unknown>) => Record<string, unknown>, note?: string) => {
    update((c) => ({ msgs: note ? [...c.msgs, { from: 'bot', blocks: [{ type: 'text', text: note }] }] : c.msgs, memory: fn(c.memory) }))
  }, [update])

  return { chat, busy, send, askOnce, patchMemory }
}
