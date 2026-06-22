import type { ReactNode } from 'react'
import { theme } from '../theme'

type Color = 'blue' | 'green' | 'yellow' | 'red' | 'gray' | 'purple'

const COLORS: Record<Color, { bg: string; border: string; text: string }> = {
  blue:   { bg: theme.accentMuted, border: theme.accent, text: theme.info },
  green:  { bg: theme.successBg, border: theme.success, text: theme.success },
  yellow: { bg: theme.warningBg, border: theme.warning, text: theme.warning },
  red:    { bg: theme.errorBg,   border: theme.error, text: theme.error },
  gray:   { bg: theme.badge,     border: theme.border, text: theme.textSecondary },
  purple: { bg: theme.accentMuted, border: theme.accent, text: theme.info },
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
