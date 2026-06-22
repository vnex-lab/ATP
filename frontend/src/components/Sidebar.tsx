import type { Page, AppStatus } from '../types'
import { theme } from '../theme'
import Logo from './Logo'
import { Check } from 'lucide-react'

interface NavItem {
  id: Page
  label: string
  step: number
  done: (s: AppStatus) => boolean
  available: (s: AppStatus) => boolean
  section?: 'studio' | 'inference'
}

const NAV: NavItem[] = [
  { id: 'data', label: 'Data', step: 1, done: (s) => s.has_training_data, available: () => true, section: 'studio' },
  { id: 'models', label: 'Library', step: 2, done: (s) => (s.saved_models_count ?? 0) > 0, available: () => true, section: 'studio' },
  { id: 'model', label: 'Architecture', step: 3, done: (s) => s.has_model, available: (s) => s.has_training_data, section: 'studio' },
  { id: 'pretrain', label: 'Pre-train', step: 4, done: () => false, available: () => true, section: 'studio' },
  { id: 'train', label: 'Train', step: 5, done: (s) => s.is_trained, available: (s) => s.has_model, section: 'studio' },
  { id: 'export', label: 'Export', step: 6, done: () => false, available: (s) => s.has_model, section: 'studio' },
  { id: 'chat', label: 'Inference', step: 1, done: () => false, available: (s) => s.is_trained, section: 'inference' },
]

interface Props {
  page: Page
  setPage: (p: Page) => void
  status: AppStatus
  compact?: boolean
}

export default function Sidebar({ page, setPage, status, compact }: Props) {
  const studioItems = NAV.filter((n) => n.section === 'studio')
  const inferenceItems = NAV.filter((n) => n.section === 'inference')

  return (
    <aside
      style={{
        width: compact ? 52 : 220,
        minWidth: compact ? 52 : 220,
        background: theme.sidebar,
        borderLeft: compact ? `1px solid ${theme.border}` : undefined,
        borderRight: compact ? undefined : `1px solid ${theme.border}`,
        display: 'flex',
        flexDirection: 'column',
        height: '100%',
        userSelect: 'none',
        order: compact ? 2 : 0,
      }}
    >
      {compact && (
        <div style={{ padding: '10px 0', display: 'flex', justifyContent: 'center', borderBottom: `1px solid ${theme.border}` }}>
          <Logo size={24} />
        </div>
      )}
      {!compact && (
        <div style={{ padding: '16px 16px 14px', borderBottom: `1px solid ${theme.border}` }}>
          <Logo size={28} showWordmark />
        </div>
      )}

      <nav style={{ flex: 1, padding: compact ? '8px 4px' : '8px 6px', display: 'flex', flexDirection: 'column', gap: 1 }}>
        {!compact && (
          <SectionLabel>Studio</SectionLabel>
        )}
        {studioItems.map((item) => (
          <NavButton
            key={item.id}
            item={item}
            active={page === item.id}
            status={status}
            compact={compact}
            onClick={() => setPage(item.id)}
          />
        ))}

        {!compact && inferenceItems.length > 0 && (
          <SectionLabel style={{ marginTop: 10 }}>Inference</SectionLabel>
        )}
        {inferenceItems.map((item) => (
          <NavButton
            key={item.id}
            item={item}
            active={page === item.id}
            status={status}
            compact={compact}
            onClick={() => setPage(item.id)}
            highlight
          />
        ))}
      </nav>

      {!compact && (
        <div style={{ padding: '12px 14px', borderTop: `1px solid ${theme.border}` }}>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 5 }}>
            <StatusRow label="Data" value={status.has_training_data ? `${status.training_data_count.toLocaleString()} pairs` : 'None'} ok={status.has_training_data} />
            <StatusRow label="Library" value={`${status.saved_models_count ?? 0} saved`} ok={(status.saved_models_count ?? 0) > 0} />
            <StatusRow label="Model" value={status.active_model_name ?? (status.has_model ? (status.model_type ?? 'Ready') : 'None')} ok={status.has_model} />
            <StatusRow label="Mode" value={status.training_mode === 'finetune' ? 'Fine-tune' : 'Scratch'} ok={status.has_model} />
            <StatusRow label="CoT" value={status.cot_reasoning?.enabled ? 'On' : 'Off'} ok={!!status.cot_reasoning?.enabled} />
          </div>
          {status.training.gpu_available && (
            <div style={{ marginTop: 10, display: 'flex', alignItems: 'center', gap: 6 }}>
              <span style={{ width: 6, height: 6, borderRadius: '50%', background: theme.success, display: 'inline-block' }} />
              <span style={{ fontSize: 11, color: theme.success }}>GPU active</span>
            </div>
          )}
        </div>
      )}
    </aside>
  )
}

function SectionLabel({ children, style }: { children: React.ReactNode; style?: React.CSSProperties }) {
  return (
    <div style={{
      fontSize: 10,
      color: theme.textDim,
      fontWeight: 600,
      letterSpacing: '0.06em',
      textTransform: 'uppercase',
      padding: '6px 10px 8px',
      ...style,
    }}>
      {children}
    </div>
  )
}

function NavButton({
  item,
  active,
  status,
  compact,
  onClick,
  highlight,
}: {
  item: NavItem
  active: boolean
  status: AppStatus
  compact?: boolean
  onClick: () => void
  highlight?: boolean
}) {
  const done = item.done(status)
  const available = item.available(status)

  if (compact) {
    return (
      <button
        onClick={onClick}
        disabled={!available}
        title={item.label}
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          width: '100%',
          padding: '8px 0',
          borderRadius: 4,
          border: 'none',
          cursor: available ? 'pointer' : 'default',
          background: active ? theme.active : 'transparent',
          color: active ? '#fff' : available ? theme.textSecondary : theme.textDim,
          opacity: available ? 1 : 0.45,
        }}
      >
        <span style={{ fontSize: 11, fontWeight: 600 }}>{item.label.slice(0, 2).toUpperCase()}</span>
      </button>
    )
  }

  return (
    <button
      onClick={onClick}
      disabled={!available}
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 10,
        padding: '7px 10px',
        borderRadius: 4,
        border: 'none',
        cursor: available ? 'pointer' : 'default',
        background: active ? (highlight ? theme.accentMuted : theme.active) : 'transparent',
        color: active ? '#fff' : available ? theme.textSecondary : theme.textDim,
        fontSize: 13,
        fontWeight: active ? 500 : 400,
        textAlign: 'left',
        width: '100%',
        opacity: available ? 1 : 0.5,
      }}
      onMouseEnter={(e) => {
        if (available && !active) (e.currentTarget as HTMLButtonElement).style.background = theme.hover
      }}
      onMouseLeave={(e) => {
        if (available && !active) (e.currentTarget as HTMLButtonElement).style.background = 'transparent'
      }}
    >
      <span
        style={{
          width: 20,
          height: 20,
          borderRadius: 3,
          background: active ? theme.accent : theme.badge,
          border: `1px solid ${active ? theme.accent : theme.border}`,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          fontSize: 10,
          color: done && !active ? theme.success : active ? '#fff' : theme.textMuted,
          flexShrink: 0,
          fontFamily: theme.mono,
        }}
      >
        {done ? <Check size={11} strokeWidth={3} /> : item.step}
      </span>
      <span style={{ flex: 1 }}>{item.label}</span>
      {item.id === 'train' && status.training.is_training && (
        <span style={{ width: 6, height: 6, borderRadius: '50%', background: theme.accent, flexShrink: 0 }} className="pulse-dot" />
      )}
    </button>
  )
}

function StatusRow({ label, value, ok }: { label: string; value: string; ok: boolean }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 8 }}>
      <span style={{ fontSize: 11, color: theme.textMuted }}>{label}</span>
      <span style={{ fontSize: 11, color: ok ? theme.text : theme.textDim, textAlign: 'right', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: 110 }}>
        {value}
      </span>
    </div>
  )
}
