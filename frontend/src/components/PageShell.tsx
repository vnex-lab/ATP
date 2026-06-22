import type { ReactNode } from 'react'
import { theme } from '../theme'

interface Props {
  title: string
  subtitle?: string
  badge?: ReactNode
  children: ReactNode
}

export default function PageShell({ title, subtitle, badge, children }: Props) {
  return (
    <div style={{ padding: '28px 32px', maxWidth: 960, margin: '0 auto' }}>
      <div style={{ marginBottom: 24 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 4 }}>
          <h1 style={{ fontSize: 22, fontWeight: 700, color: theme.text, margin: 0, letterSpacing: '-0.4px' }}>
            {title}
          </h1>
          {badge}
        </div>
        {subtitle && (
          <p style={{ color: theme.textMuted, fontSize: 13, margin: 0, lineHeight: 1.5 }}>{subtitle}</p>
        )}
      </div>
      {children}
    </div>
  )
}
