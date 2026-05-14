import type { ReactNode } from 'react'

type Color = 'blue' | 'green' | 'yellow' | 'red' | 'gray' | 'purple'

const COLORS: Record<Color, { bg: string; border: string; text: string }> = {
  blue:   { bg: '#1e3a5f22', border: '#1e40af', text: '#93c5fd' },
  green:  { bg: '#14532d22', border: '#166534', text: '#86efac' },
  yellow: { bg: '#451a0322', border: '#92400e', text: '#fcd34d' },
  red:    { bg: '#450a0a',   border: '#7f1d1d', text: '#fca5a5' },
  gray:   { bg: '#1c1c1c',   border: '#252525', text: '#94a3b8' },
  purple: { bg: '#1e1b4b',   border: '#3730a3', text: '#a5b4fc' },
}

export default function Badge({ children, color = 'gray' }: { children: ReactNode; color?: Color }) {
  const c = COLORS[color]
  return (
    <span style={{
      fontSize: 11,
      fontWeight: 600,
      padding: '3px 9px',
      borderRadius: 20,
      background: c.bg,
      border: `1px solid ${c.border}`,
      color: c.text,
      letterSpacing: '0.02em',
    }}>
      {children}
    </span>
  )
}
