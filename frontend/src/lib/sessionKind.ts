import type { SessionSummary } from '../api'

export type SessionKind = 'Analyses' | 'Comparisons' | 'Conversations'
export const SESSION_TABS: { label: string; kind: SessionKind | null }[] = [
  { label: 'All', kind: null },
  { label: 'Analyses', kind: 'Analyses' },
  { label: 'Comparisons', kind: 'Comparisons' },
  { label: 'Conversations', kind: 'Conversations' },
]

/** A saved session is an AI conversation, a comparison (two scenarios) or a single analysis. */
export const kindOf = (s: SessionSummary): SessionKind =>
  s.kind === 'chat' ? 'Conversations' : s.mode === 'comparison' || s.mode === 'full' ? 'Comparisons' : 'Analyses'

export const singular: Record<SessionKind, string> = { Analyses: 'Analysis', Comparisons: 'Comparison', Conversations: 'Conversation' }
