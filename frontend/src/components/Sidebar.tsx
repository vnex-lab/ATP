import type { Page, AppStatus } from '../types'

interface NavItem {
  id: Page
  label: string
  icon: string
  step: number
  done: (s: AppStatus) => boolean
  available: (s: AppStatus) => boolean
}

const NAV: NavItem[] = [
  {
    id: 'data',
    label: 'Data',
    icon: '⬆',
    step: 1,
    done: (s) => s.has_training_data,
    available: () => true,
  },
  {
    id: 'model',
    label: 'Model',
    icon: '⬡',
    step: 2,
    done: (s) => s.has_model,
    available: (s) => s.has_training_data,
  },
  {
    id: 'train',
    label: 'Train',
    icon: '▶',
    step: 3,
    done: (s) => s.is_trained,
    available: (s) => s.has_model,
  },
  {
    id: 'chat',
    label: 'Chat',
    icon: '◎',
    step: 4,
    done: () => false,
    available: (s) => s.is_trained,
  },
  {
    id: 'export',
    label: 'Export',
    icon: '⬇',
    step: 5,
    done: () => false,
    available: (s) => s.has_model,
  },
]

interface Props {
  page: Page
  setPage: (p: Page) => void
  status: AppStatus
}

export default function Sidebar({ page, setPage, status }: Props) {
  return (
    <aside
      style={{
        width: 220,
        minWidth: 220,
        background: '#0f0f0f',
        borderRight: '1px solid #1e1e1e',
        display: 'flex',
        flexDirection: 'column',
        height: '100%',
        userSelect: 'none',
      }}
    >
      {/* Logo */}
      <div style={{ padding: '20px 20px 16px', borderBottom: '1px solid #1e1e1e' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <div
            style={{
              width: 32,
              height: 32,
              borderRadius: 8,
              background: 'linear-gradient(135deg, #3b82f6, #6366f1)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontSize: 16,
              flexShrink: 0,
            }}
          >
            🤖
          </div>
          <div>
            <div style={{ fontWeight: 700, fontSize: 15, color: '#e2e8f0', letterSpacing: '-0.3px' }}>
              LLM Vite
            </div>
            <div style={{ fontSize: 11, color: '#475569', marginTop: 1 }}>
              Train your own AI
            </div>
          </div>
        </div>
      </div>

      {/* Nav */}
      <nav style={{ flex: 1, padding: '12px 10px', display: 'flex', flexDirection: 'column', gap: 2 }}>
        <div style={{ fontSize: 10, color: '#374151', fontWeight: 600, letterSpacing: '0.08em', textTransform: 'uppercase', padding: '6px 10px 8px' }}>
          Workflow
        </div>
        {NAV.map((item) => {
          const active = page === item.id
          const done = item.done(status)
          const available = item.available(status)
          return (
            <button
              key={item.id}
              onClick={() => setPage(item.id)}
              disabled={!available}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 10,
                padding: '8px 10px',
                borderRadius: 7,
                border: 'none',
                cursor: available ? 'pointer' : 'default',
                background: active ? '#1e3a5f' : 'transparent',
                color: active ? '#93c5fd' : available ? '#94a3b8' : '#374151',
                fontSize: 13,
                fontWeight: active ? 600 : 400,
                textAlign: 'left',
                width: '100%',
                transition: 'all 0.1s',
                opacity: available ? 1 : 0.45,
              }}
              onMouseEnter={(e) => {
                if (available && !active) {
                  ;(e.currentTarget as HTMLButtonElement).style.background = '#151515'
                  ;(e.currentTarget as HTMLButtonElement).style.color = '#cbd5e1'
                }
              }}
              onMouseLeave={(e) => {
                if (available && !active) {
                  ;(e.currentTarget as HTMLButtonElement).style.background = 'transparent'
                  ;(e.currentTarget as HTMLButtonElement).style.color = available ? '#94a3b8' : '#374151'
                }
              }}
            >
              <span
                style={{
                  width: 24,
                  height: 24,
                  borderRadius: 6,
                  background: active ? '#2563eb' : done ? '#14532d' : '#1c1c1c',
                  border: `1px solid ${active ? '#3b82f6' : done ? '#16a34a' : '#252525'}`,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  fontSize: 10,
                  color: active ? '#fff' : done ? '#22c55e' : '#64748b',
                  flexShrink: 0,
                  fontFamily: 'monospace',
                }}
              >
                {done ? '✓' : item.step}
              </span>
              <span style={{ flex: 1 }}>{item.label}</span>
              {item.id === 'train' && status.training.is_training && (
                <span
                  className="pulse-dot"
                  style={{
                    width: 6,
                    height: 6,
                    borderRadius: '50%',
                    background: '#3b82f6',
                    flexShrink: 0,
                  }}
                />
              )}
            </button>
          )
        })}
      </nav>

      {/* Status bar */}
      <div style={{ padding: '12px 16px', borderTop: '1px solid #1e1e1e' }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
          <StatusRow
            label="Data"
            value={status.has_training_data ? `${status.training_data_count.toLocaleString()} pairs` : 'None'}
            ok={status.has_training_data}
          />
          <StatusRow
            label="Vocab"
            value={status.has_tokenizer ? `${status.tokenizer_vocab_size.toLocaleString()} tokens` : 'None'}
            ok={status.has_tokenizer}
          />
          <StatusRow
            label="Model"
            value={status.has_model ? (status.model_type ?? 'Ready') : 'None'}
            ok={status.has_model}
          />
          <StatusRow
            label="Trained"
            value={status.is_trained ? 'Yes' : 'No'}
            ok={status.is_trained}
          />
        </div>
        {status.training.gpu_available && (
          <div style={{ marginTop: 10, display: 'flex', alignItems: 'center', gap: 6 }}>
            <span style={{ width: 6, height: 6, borderRadius: '50%', background: '#22c55e', display: 'inline-block' }} />
            <span style={{ fontSize: 11, color: '#22c55e' }}>GPU Active</span>
          </div>
        )}
        <div style={{ marginTop: 10, fontSize: 10, color: '#374151' }}>
          LLM Vite · NumPy/CuPy
        </div>
      </div>
    </aside>
  )
}

function StatusRow({ label, value, ok }: { label: string; value: string; ok: boolean }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
      <span style={{ fontSize: 11, color: '#475569' }}>{label}</span>
      <span style={{ fontSize: 11, color: ok ? '#22c55e' : '#475569', fontWeight: ok ? 500 : 400 }}>
        {value}
      </span>
    </div>
  )
}
