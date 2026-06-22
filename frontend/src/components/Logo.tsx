import { theme } from '../theme'

interface Props {
  size?: number
  showWordmark?: boolean
}

/** Model Studio mark — layered blocks suggesting model weights / stack. */
export default function Logo({ size = 28, showWordmark = false }: Props) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
      <svg
        width={size}
        height={size}
        viewBox="0 0 32 32"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        aria-hidden
      >
        <rect width="32" height="32" rx="7" fill={theme.accent} />
        <rect x="7" y="7" width="18" height="4" rx="1.2" fill="white" fillOpacity="0.95" />
        <rect x="7" y="14" width="18" height="4" rx="1.2" fill="white" fillOpacity="0.72" />
        <rect x="7" y="21" width="18" height="4" rx="1.2" fill="white" fillOpacity="0.48" />
        <circle cx="24" cy="9" r="2" fill={theme.success} />
      </svg>
      {showWordmark && (
        <div>
          <div style={{ fontWeight: 600, fontSize: 13, color: theme.text, lineHeight: 1.2 }}>Model Studio</div>
          <div style={{ fontSize: 10, color: theme.textMuted, marginTop: 1 }}>Train and deploy models</div>
        </div>
      )}
    </div>
  )
}

export function logoDataUri(): string {
  return "data:image/svg+xml," + encodeURIComponent(
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32"><rect width="32" height="32" rx="7" fill="%23007acc"/><rect x="7" y="7" width="18" height="4" rx="1.2" fill="white" fill-opacity="0.95"/><rect x="7" y="14" width="18" height="4" rx="1.2" fill="white" fill-opacity="0.72"/><rect x="7" y="21" width="18" height="4" rx="1.2" fill="white" fill-opacity="0.48"/><circle cx="24" cy="9" r="2" fill="%234ec9b0"/></svg>'
  )
}
