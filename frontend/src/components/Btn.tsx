import type { ReactNode } from 'react'

interface Props {
  children: ReactNode
  onClick?: () => void
  variant?: 'default' | 'primary' | 'danger'
  disabled?: boolean
  type?: 'button' | 'submit'
}

export default function Btn({ children, onClick, variant = 'default', disabled, type = 'button' }: Props) {
  const base: React.CSSProperties = {
    padding: '8px 16px',
    borderRadius: 7,
    border: '1px solid',
    fontSize: 13,
    fontWeight: 500,
    cursor: disabled ? 'default' : 'pointer',
    fontFamily: 'inherit',
    transition: 'all 0.1s',
    opacity: disabled ? 0.5 : 1,
    outline: 'none',
    whiteSpace: 'nowrap',
  }

  const variants: Record<string, React.CSSProperties> = {
    default: { background: '#1c1c1c', borderColor: '#252525', color: '#94a3b8' },
    primary: { background: '#2563eb', borderColor: '#2563eb', color: '#fff' },
    danger: { background: '#7f1d1d', borderColor: '#991b1b', color: '#fca5a5' },
  }

  return (
    <button type={type} onClick={disabled ? undefined : onClick} style={{ ...base, ...variants[variant] }}>
      {children}
    </button>
  )
}
