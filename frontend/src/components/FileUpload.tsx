import { useRef, useState } from 'react'
import { Upload, X } from 'lucide-react'
import { theme } from '../theme'

interface Props {
  label: string
  accept?: string
  optional?: boolean
  disabled?: boolean
  onFileChange: (file: File | null) => void
}

export default function FileUpload({ label, accept, optional, disabled, onFileChange }: Props) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [fileName, setFileName] = useState<string | null>(null)

  const pick = () => {
    if (!disabled) inputRef.current?.click()
  }

  const clear = (e: React.MouseEvent) => {
    e.stopPropagation()
    setFileName(null)
    onFileChange(null)
    if (inputRef.current) inputRef.current.value = ''
  }

  return (
    <div>
      <div style={{ fontSize: 12, color: theme.textSecondary, marginBottom: 6 }}>
        {label}
        {optional && <span style={{ color: theme.textDim, marginLeft: 6 }}>(optional)</span>}
      </div>
      <button
        type="button"
        onClick={pick}
        disabled={disabled}
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 10,
          width: '100%',
          padding: '9px 12px',
          borderRadius: 4,
          border: `1px solid ${fileName ? theme.accent : theme.border}`,
          background: theme.input,
          color: fileName ? theme.text : theme.textMuted,
          fontSize: 13,
          cursor: disabled ? 'default' : 'pointer',
          fontFamily: theme.font,
          textAlign: 'left',
          opacity: disabled ? 0.5 : 1,
        }}
      >
        <Upload size={15} style={{ flexShrink: 0, color: theme.accent }} />
        <span style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
          {fileName ?? 'Choose file...'}
        </span>
        {fileName && (
          <span role="button" tabIndex={0} onClick={clear} style={{ display: 'flex', color: theme.textMuted }}>
            <X size={14} />
          </span>
        )}
      </button>
      <input
        ref={inputRef}
        type="file"
        accept={accept}
        style={{ display: 'none' }}
        onChange={(e) => {
          const f = e.target.files?.[0] ?? null
          setFileName(f?.name ?? null)
          onFileChange(f)
        }}
      />
    </div>
  )
}
