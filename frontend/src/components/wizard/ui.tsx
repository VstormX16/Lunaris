import { ArrowLeft, ArrowRight } from 'lucide-react'
import type { ReactNode } from 'react'

export const inputCls =
  'w-full rounded-[8px] border border-line bg-white/70 px-3 py-2.5 text-[17px] text-ink outline-none transition focus:border-brand placeholder:text-muted/70'

/** Primary action: 4px frosted rim around a crisp white button (same as the home trigger). */
export function GlassButton({
  children,
  onClick,
  disabled,
}: {
  children: ReactNode
  onClick?: () => void
  disabled?: boolean
}) {
  return (
    <span className={`glass-rim inline-flex rounded-[10px] p-[4px] transition-opacity ${disabled ? 'opacity-50' : ''}`}>
      <button
        onClick={onClick}
        disabled={disabled}
        className="gem-border grain group flex items-center gap-3 rounded-[8px] border border-brand bg-white px-7 py-2.5 text-[18px] text-ink transition duration-200 hover:text-brand hover:shadow-[0_6px_16px_-8px_rgba(38,70,60,0.45)] active:scale-[0.98] disabled:cursor-not-allowed disabled:border-line disabled:text-muted disabled:shadow-none"
      >
        {children} <ArrowRight size={18} className="text-brand transition-transform duration-200 group-hover:translate-x-0.5 group-disabled:translate-x-0" />
      </button>
    </span>
  )
}

/** Bottom bar of every step: quiet Back on the left, primary action on the right. */
export function StepFooter({
  onBack,
  children,
}: {
  onBack?: () => void
  children: ReactNode
}) {
  return (
    <div className="mt-8 flex items-center justify-between border-t border-line pt-5">
      {onBack ? (
        <button onClick={onBack} className="flex items-center gap-2 text-[15px] text-muted hover:text-ink">
          <ArrowLeft size={16} /> Back
        </button>
      ) : (
        <span />
      )}
      {children}
    </div>
  )
}
