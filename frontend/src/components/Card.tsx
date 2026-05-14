import type { ReactNode, CSSProperties } from 'react'

export default function Card({ children, style }: { children: ReactNode; style?: CSSProperties }) {
  return (
    <div style={{
      background: '#0f0f0f',
      border: '1px solid #1e1e1e',
      borderRadius: 10,
      padding: '18px 20px',
      ...style,
    }}>
      {children}
    </div>
  )
}
