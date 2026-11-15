import { ImageOff } from 'lucide-react'
import { useState } from 'react'

/**
 * Vehicle photo from public/spacecraft/<shape>/<id>.jpg; falls back to an icon when the file is missing.
 * `square` is for tiles and thumbnails, `wide` (16:9) for the detail page, so no picture is ever cropped to fit.
 */
export default function Photo({ id, name, shape = 'square', className = '', iconSize = 24 }: { id: string; name: string; shape?: 'square' | 'wide'; className?: string; iconSize?: number }) {
  const [missing, setMissing] = useState(false)
  return (
    <div className={`flex items-center justify-center overflow-hidden bg-active text-muted/50 ${className}`}>
      {missing ? (
        <ImageOff size={iconSize} aria-label="Photo not added yet" />
      ) : (
        <img src={`/spacecraft/${shape}/${id}.jpg`} alt={name} loading="lazy" onError={() => setMissing(true)} className="h-full w-full object-cover" />
      )}
    </div>
  )
}
