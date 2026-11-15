import type { AssistantBlock } from '../api'

export type Msg = { from: 'user'; text: string } | { from: 'bot'; blocks: AssistantBlock[] }
export type Chat = { msgs: Msg[]; memory: Record<string, unknown> }

export const emptyChat: Chat = { msgs: [], memory: {} }

/** A chat is saved as one list: the messages, then one last entry that holds the assistant's memory of the conversation. */
export const toStored = (c: Chat): unknown[] => [...c.msgs, { from: 'memory', memory: c.memory }]

export function fromStored(list: unknown[] | undefined): Chat {
  const msgs: Msg[] = []
  let memory: Record<string, unknown> = {}
  for (const m of list ?? []) {
    const o = m as { from?: string; text?: unknown; blocks?: unknown; memory?: unknown }
    if (o?.from === 'user' && typeof o.text === 'string') msgs.push({ from: 'user', text: o.text })
    else if (o?.from === 'bot' && Array.isArray(o.blocks)) msgs.push({ from: 'bot', blocks: o.blocks as AssistantBlock[] })
    else if (o?.from === 'memory' && o.memory && typeof o.memory === 'object') memory = o.memory as Record<string, unknown>
  }
  return { msgs, memory }
}
