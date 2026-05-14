import type { ReactNode } from 'react'

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
          <h1 style={{ fontSize: 22, fontWeight: 700, color: '#e2e8f0', margin: 0, letterSpacing: '-0.4px' }}>
            {title}
          </h1>
          {badge}
        </div>
        {subtitle && (
          <p style={{ color: '#64748b', fontSize: 13, margin: 0, lineHeight: 1.5 }}>{subtitle}</p>
        )}
      </div>
      {children}
    </div>
  )
}
