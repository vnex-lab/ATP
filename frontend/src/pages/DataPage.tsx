import { useState, useRef, useCallback } from 'react'
import type { AppStatus } from '../types'
import { uploadFile, uploadText, loadBuiltinAssistant, loadBuiltinSFT, getDataInfo } from '../api'
import PageShell from '../components/PageShell'
import Card from '../components/Card'
import Btn from '../components/Btn'
import Badge from '../components/Badge'
import { theme } from '../theme'
import { textareaStyle, labelStyle, tabStyle, dropZoneStyle, sectionLabel, alertColors, insetBox } from '../styles'

type Method = 'file' | 'text' | 'assistant' | 'sft'

export default function DataPage({ status, onRefresh }: { status: AppStatus; onRefresh: () => void }) {
  const [method, setMethod] = useState<Method>('file')
  const [loading, setLoading] = useState(false)
  const [msg, setMsg] = useState<{ type: 'success' | 'error' | 'info'; text: string } | null>(null)
  const [preview, setPreview] = useState<{ user: string; bot: string }[]>([])

  // File upload state
  const fileRef = useRef<HTMLInputElement>(null)
  const [dragging, setDragging] = useState(false)

  // Text entry state
  const [pairs, setPairs] = useState<{ user: string; bot: string }[]>([])
  const [newUser, setNewUser] = useState('')
  const [newBot, setNewBot] = useState('')

  // Built-in state
  const [targetMb, setTargetMb] = useState(90)
  const [sftRows, setSftRows] = useState(25000)

  const ok = (text: string, p?: { user: string; bot: string }[]) => {
    setMsg({ type: 'success', text })
    if (p) setPreview(p)
    onRefresh()
  }
  const err = (text: string) => setMsg({ type: 'error', text })

  const handleFile = useCallback(async (file: File) => {
    setLoading(true)
    setMsg(null)
    try {
      const res = await uploadFile(file) as { count: number; preview: { user: string; bot: string }[] }
      ok(`Loaded ${res.count.toLocaleString()} conversation pairs from "${file.name}"`, res.preview)
    } catch (e: unknown) {
      err((e as Error).message)
    } finally {
      setLoading(false)
    }
  }, [])

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setDragging(false)
    const file = e.dataTransfer.files[0]
    if (file) handleFile(file)
  }

  const handleTextUpload = async () => {
    if (pairs.length === 0) return err('Add at least one pair.')
    setLoading(true)
    setMsg(null)
    try {
      const res = await uploadText(pairs) as { count: number }
      ok(`Loaded ${res.count} manually entered pairs.`, pairs.slice(0, 3))
    } catch (e: unknown) {
      err((e as Error).message)
    } finally {
      setLoading(false)
    }
  }

  const handleBuiltin = async (type: 'assistant' | 'sft') => {
    setLoading(true)
    setMsg({ type: 'info', text: 'Generating dataset… this may take a few seconds.' })
    try {
      let res: { count: number; mb: number }
      if (type === 'assistant') {
        res = await loadBuiltinAssistant(targetMb) as { count: number; mb: number }
      } else {
        res = await loadBuiltinSFT(sftRows) as { count: number; mb: number }
      }
      ok(`Loaded ${res.count.toLocaleString()} pairs (~${res.mb} MB).`)
      const info = await getDataInfo() as { preview: { user: string; bot: string }[] }
      setPreview(info.preview)
    } catch (e: unknown) {
      err((e as Error).message)
    } finally {
      setLoading(false)
    }
  }

  const METHODS: { id: Method; label: string; desc: string }[] = [
    { id: 'file', label: 'Upload File', desc: 'JSON, JSONL, CSV, TSV, TXT, Parquet' },
    { id: 'text', label: 'Enter Manually', desc: 'Type pairs directly' },
    { id: 'assistant', label: 'Built-in Assistant', desc: 'Large general-purpose dataset' },
    { id: 'sft', label: 'Built-in SFT Pack', desc: 'Instruction-following dataset' },
  ]

  return (
    <PageShell
      title="Training Data"
      subtitle="Upload conversation pairs to train your model."
      badge={status.has_training_data ? <Badge color="green">{status.training_data_count.toLocaleString()} pairs loaded</Badge> : undefined}
    >
      {/* Method tabs */}
      <div style={{ display: 'flex', gap: 6, marginBottom: 20, flexWrap: 'wrap' }}>
        {METHODS.map((m) => (
          <button
            key={m.id}
            onClick={() => { setMethod(m.id); setMsg(null) }}
            style={tabStyle(method === m.id)}
          >
            {m.label}
          </button>
        ))}
      </div>

      {msg && <Alert type={msg.type} text={msg.text} />}

      {/* File upload */}
      {method === 'file' && (
        <Card>
          <div
            onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
            onDragLeave={() => setDragging(false)}
            onDrop={handleDrop}
            onClick={() => fileRef.current?.click()}
            style={dropZoneStyle(dragging)}
          >
            <div style={{ fontSize: 28, marginBottom: 10, color: theme.textMuted }}>+</div>
            <div style={{ color: theme.text, fontWeight: 500, marginBottom: 6 }}>
              Drop a file here or click to browse
            </div>
            <div style={{ color: theme.textMuted, fontSize: 12 }}>
              Supports: JSON · JSONL · CSV · TSV · TXT · Parquet
            </div>
            <input
              ref={fileRef}
              type="file"
              style={{ display: 'none' }}
              accept=".json,.jsonl,.csv,.tsv,.txt,.parquet"
              onChange={(e) => { const f = e.target.files?.[0]; if (f) handleFile(f) }}
            />
          </div>
          {loading && <Spinner text="Parsing file…" />}
          <div style={{ marginTop: 16 }}>
            <SectionLabel>Supported Formats</SectionLabel>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, marginTop: 8 }}>
              {[
                ['JSON/JSONL', '{"user": "...", "bot": "..."} per entry'],
                ['CSV/TSV', 'Columns: user/bot, question/answer, etc.'],
                ['Text', 'Lines with | → - : tab separator'],
                ['Parquet', 'HuggingFace datasets, The Stack, etc.'],
                ['Code Debug', '{"original_src": ..., "changed_src": ...}'],
              ].map(([fmt, desc]) => (
                <div key={fmt} style={{ ...insetBox, padding: '8px 12px' }}>
                  <div style={{ color: theme.info, fontSize: 12, fontWeight: 600, fontFamily: theme.mono }}>{fmt}</div>
                  <div style={{ color: theme.textMuted, fontSize: 11, marginTop: 2 }}>{desc}</div>
                </div>
              ))}
            </div>
          </div>
        </Card>
      )}

      {/* Manual text entry */}
      {method === 'text' && (
        <Card>
          <SectionLabel>Add conversation pairs</SectionLabel>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginTop: 10 }}>
            <div>
              <label style={labelStyle}>User message</label>
              <textarea
                value={newUser}
                onChange={(e) => setNewUser(e.target.value)}
                placeholder="Hello, how are you?"
                style={textareaStyle}
                rows={3}
              />
            </div>
            <div>
              <label style={labelStyle}>Bot response</label>
              <textarea
                value={newBot}
                onChange={(e) => setNewBot(e.target.value)}
                placeholder="I'm doing great, thanks!"
                style={textareaStyle}
                rows={3}
              />
            </div>
          </div>
          <div style={{ display: 'flex', gap: 10, marginTop: 12 }}>
            <Btn
              onClick={() => {
                if (newUser.trim() && newBot.trim()) {
                  setPairs([...pairs, { user: newUser.trim(), bot: newBot.trim() }])
                  setNewUser('')
                  setNewBot('')
                }
              }}
            >
              + Add Pair
            </Btn>
            <Btn variant="primary" onClick={handleTextUpload} disabled={pairs.length === 0 || loading}>
              Use These {pairs.length > 0 ? `(${pairs.length})` : ''} Pairs
            </Btn>
          </div>
          {pairs.length > 0 && (
            <div style={{ marginTop: 16, maxHeight: 240, overflowY: 'auto' }}>
              <SectionLabel>{pairs.length} pairs</SectionLabel>
              {pairs.map((p, i) => (
                <div key={i} style={{ display: 'flex', alignItems: 'flex-start', gap: 10, padding: '8px 0', borderBottom: '1px solid #1e1e1e' }}>
                  <div style={{ flex: 1 }}>
                    <span style={{ color: '#3b82f6', fontSize: 11, fontWeight: 600 }}>USER</span>
                    <div style={{ color: '#e2e8f0', fontSize: 13, marginTop: 2 }}>{p.user}</div>
                  </div>
                  <div style={{ flex: 1 }}>
                    <span style={{ color: '#22c55e', fontSize: 11, fontWeight: 600 }}>BOT</span>
                    <div style={{ color: '#e2e8f0', fontSize: 13, marginTop: 2 }}>{p.bot}</div>
                  </div>
                  <button
                    onClick={() => setPairs(pairs.filter((_, j) => j !== i))}
                    style={{ background: 'none', border: 'none', color: '#475569', cursor: 'pointer', fontSize: 16, padding: 4 }}
                  >×</button>
                </div>
              ))}
            </div>
          )}
        </Card>
      )}

      {/* Built-in assistant */}
      {method === 'assistant' && (
        <Card>
          <SectionLabel>Built-in Assistant Dialog Dataset</SectionLabel>
          <p style={{ color: theme.textSecondary, fontSize: 13, margin: '8px 0 16px' }}>
            A large collection of general assistant-style conversation pairs. Ideal for training a conversational model from scratch.
          </p>
          <div style={{ marginBottom: 16 }}>
            <label style={labelStyle}>Target dataset size: <strong style={{ color: theme.text }}>{targetMb} MB</strong></label>
            <input
              type="range"
              min={15}
              max={100}
              value={targetMb}
              onChange={(e) => setTargetMb(Number(e.target.value))}
              style={{ width: '100%', marginTop: 8, accentColor: theme.accent }}
            />
            <div style={{ display: 'flex', justifyContent: 'space-between', color: '#475569', fontSize: 11, marginTop: 4 }}>
              <span>15 MB (fast)</span>
              <span>100 MB (comprehensive)</span>
            </div>
          </div>
          <Btn variant="primary" onClick={() => handleBuiltin('assistant')} disabled={loading}>
            {loading ? 'Generating…' : 'Generate & Load Dataset'}
          </Btn>
        </Card>
      )}

      {/* Built-in SFT */}
      {method === 'sft' && (
        <Card>
          <SectionLabel>Built-in SFT Pack</SectionLabel>
          <p style={{ color: '#94a3b8', fontSize: 13, margin: '8px 0 4px' }}>
            Instruction → response pairs optimized for <strong style={{ color: '#e2e8f0' }}>Decoder-Only</strong> Transformer training with SFT mode enabled.
          </p>
          <div style={{ background: '#1c2a1c', border: '1px solid #1a3a1a', borderRadius: 7, padding: '8px 12px', marginBottom: 16, fontSize: 12, color: '#86efac' }}>
            Best used with: Transformer Decoder-Only + SFT Loss enabled in Training
          </div>
          <div style={{ marginBottom: 16 }}>
            <label style={labelStyle}>Number of pairs: <strong style={{ color: '#e2e8f0' }}>{sftRows.toLocaleString()}</strong></label>
            <input
              type="range"
              min={3000}
              max={100000}
              step={1000}
              value={sftRows}
              onChange={(e) => setSftRows(Number(e.target.value))}
              style={{ width: '100%', marginTop: 8, accentColor: theme.accent }}
            />
            <div style={{ display: 'flex', justifyContent: 'space-between', color: '#475569', fontSize: 11, marginTop: 4 }}>
              <span>3,000 (fast)</span>
              <span>100,000 (thorough)</span>
            </div>
          </div>
          <Btn variant="primary" onClick={() => handleBuiltin('sft')} disabled={loading}>
            {loading ? 'Generating…' : 'Generate & Load SFT Pack'}
          </Btn>
        </Card>
      )}

      {/* Preview */}
      {(preview.length > 0 || status.has_training_data) && (
        <div style={{ marginTop: 20 }}>
          <PreviewSection preview={preview} status={status} />
        </div>
      )}
    </PageShell>
  )
}

function PreviewSection({ preview, status }: { preview: { user: string; bot: string }[]; status: AppStatus }) {
  const items = preview.length > 0 ? preview : []
  if (items.length === 0 && !status.has_training_data) return null
  return (
    <Card>
      <SectionLabel>Data Preview</SectionLabel>
      {status.has_training_data && (
        <div style={{ display: 'flex', gap: 12, margin: '8px 0 14px', flexWrap: 'wrap' }}>
          <Stat label="Total Pairs" value={status.training_data_count.toLocaleString()} />
          {status.training_data_profile && <Stat label="Profile" value={status.training_data_profile} />}
        </div>
      )}
      {items.length > 0 && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
          {items.slice(0, 3).map((p, i) => (
            <div key={i} style={{ ...insetBox, padding: 12 }}>
              <div style={{ display: 'flex', gap: 8, marginBottom: 6 }}>
                <span style={{ fontSize: 11, color: theme.accent, fontWeight: 700, background: theme.accentMuted, padding: '2px 6px', borderRadius: 4 }}>USER</span>
              </div>
              <div style={{ color: theme.text, fontSize: 13, marginBottom: 10, fontStyle: 'italic', whiteSpace: 'pre-wrap' }}>
                {p.user.length > 200 ? p.user.slice(0, 200) + '…' : p.user}
              </div>
              <div style={{ display: 'flex', gap: 8, marginBottom: 6 }}>
                <span style={{ fontSize: 11, color: '#22c55e', fontWeight: 700, background: '#14532d', padding: '2px 6px', borderRadius: 4 }}>BOT</span>
              </div>
              <div style={{ color: '#cbd5e1', fontSize: 13, whiteSpace: 'pre-wrap' }}>
                {p.bot.length > 200 ? p.bot.slice(0, 200) + '…' : p.bot}
              </div>
            </div>
          ))}
        </div>
      )}
    </Card>
  )
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div style={{ ...insetBox, padding: '6px 12px' }}>
      <div style={{ fontSize: 11, color: theme.textMuted }}>{label}</div>
      <div style={{ fontSize: 14, color: theme.text, fontWeight: 600 }}>{value}</div>
    </div>
  )
}

function Alert({ type, text }: { type: 'success' | 'error' | 'info'; text: string }) {
  const c = alertColors[type]
  return (
    <div style={{ background: c.bg, border: `1px solid ${c.border}`, borderRadius: 6, padding: '10px 14px', marginBottom: 16, color: c.text, fontSize: 13 }}>
      {text}
    </div>
  )
}

function Spinner({ text }: { text: string }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '12px 0', color: theme.textSecondary, fontSize: 13 }}>
      <span className="pulse-dot" style={{ width: 8, height: 8, borderRadius: '50%', background: theme.accent, display: 'inline-block' }} />
      {text}
    </div>
  )
}

function SectionLabel({ children }: { children: React.ReactNode }) {
  return <div style={sectionLabel}>{children}</div>
}
