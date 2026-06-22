import type { ReactNode } from 'react'
import { theme } from '../theme'

interface Props {
  children: ReactNode
  onClick?: () => void
  variant?: 'default' | 'primary' | 'danger'
  disabled?: boolean
  type?: 'button' | 'submit'
}

export default function Btn({ children, onClick, variant = 'default', disabled, type = 'button' }: Props) {
  const base: React.CSSProperties = {
    padding: '6px 14px',
    borderRadius: 4,
    border: '1px solid',
    fontSize: 13,
    fontWeight: 400,
    cursor: disabled ? 'default' : 'pointer',
    fontFamily: theme.font,
    opacity: disabled ? 0.5 : 1,
    outline: 'none',
    whiteSpace: 'nowrap',
  }

  const variants: Record<string, React.CSSProperties> = {
    default: { background: theme.panel, borderColor: theme.border, color: theme.text },
    primary: { background: theme.accent, borderColor: theme.accent, color: '#fff' },
    danger: { background: theme.errorBg, borderColor: theme.error, color: theme.error },
  }

  return (
    <button type={type} onClick={disabled ? undefined : onClick} style={{ ...base, ...variants[variant] }}>
      {children}
    </button>
  )
}
