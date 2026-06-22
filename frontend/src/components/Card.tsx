import type { ReactNode, CSSProperties } from 'react'
import { theme } from '../theme'

export default function Card({ children, style }: { children: ReactNode; style?: CSSProperties }) {
  return (
    <div style={{
      background: theme.panel,
      border: `1px solid ${theme.border}`,
      borderRadius: 6,
      padding: '16px 18px',
      ...style,
    }}>
      {children}
    </div>
  )
}
