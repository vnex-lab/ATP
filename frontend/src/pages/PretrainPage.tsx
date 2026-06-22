import { useState, useEffect } from 'react'
import type { AppStatus, PretrainStatus } from '../types'
import { getPretrainStatus, loadPretrain, deletePretrain, getCotStatus, loadCotBuiltin } from '../api'
import PageShell from '../components/PageShell'
import Card from '../components/Card'
import Btn from '../components/Btn'
import { theme } from '../theme'
import { insetBox } from '../styles'

export default function PretrainPage({ status, onRefresh }: { status: AppStatus; onRefresh: () => void }) {
  const [ptStatus, setPtStatus] = useState<PretrainStatus | null>(null)
  const [loading, setLoading] = useState(false)
  const [msg, setMsg] = useState<{ type: 'success' | 'error' | 'info'; text: string } | null>(null)

  const refresh = async () => {
    try {
      const s = await getPretrainStatus()
      setPtStatus(s)
    } catch {}
  }

  useEffect(() => { refresh() }, [])

  const handleLoad = async () => {
    setLoading(true)
    setMsg(null)
    try {
      const res = await loadPretrain() as { count: number; message: string }
      setMsg({ type: 'success', text: res.message })
      await refresh()
    } catch (e: unknown) {
      setMsg({ type: 'error', text: (e as Error).message })
    } finally {
      setLoading(false)
    }
  }

  const handleDelete = async () => {
    setLoading(true)
    setMsg(null)
    try {
      await deletePretrain()
      setMsg({ type: 'info', text: 'Pre-training data deleted from disk and memory.' })
      await refresh()
    } catch (e: unknown) {
      setMsg({ type: 'error', text: (e as Error).message })
    } finally {
      setLoading(false)
    }
  }

  const loaded = ptStatus?.loaded ?? false
  const count  = ptStatus?.count  ?? 0

  return (
    <PageShell
      title="Pre-training"
      subtitle="Load a foundation dataset to give your model basic language understanding before fine-tuning."
    >
      {msg && <Alert type={msg.type} text={msg.text} />}

      {/* What is pre-training */}
      <Card style={{ marginBottom: 16 }}>
        <div style={{ fontWeight: 600, fontSize: 14, color: theme.text, marginBottom: 10 }}>What is Pre-training?</div>
        <p style={{ color: theme.textSecondary, fontSize: 13, lineHeight: 1.7, margin: '0 0 12px' }}>
          Pre-training teaches your model fundamental language patterns — greetings, politeness, basic
          question-answer structure — before you fine-tune it on your specific data. Without this, a model
          trained on only a few custom pairs often produces word salad because it has no grasp of how language
          flows at all.
        </p>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
          {[
            ['Without pre-train', 'Model sees only your pairs. Small datasets produce incoherent outputs.'],
            ['With pre-train', 'Model first learns language basics, then your fine-tune pairs snap into place.'],
            ['Persists on disk', 'Pre-train data is saved to pretrain_data.json. It survives restarts.'],
            ['Delete to reset', 'Click Delete to remove it. The only way to clear it.'],
          ].map(([title, desc]) => (
            <div key={title} style={{ background: theme.panel, borderRadius: 8, padding: '10px 14px', border: `1px solid ${theme.border}` }}>
              <div style={{ color: '#93c5fd', fontSize: 12, fontWeight: 600, marginBottom: 4 }}>{title}</div>
              <div style={{ color: '#475569', fontSize: 12, lineHeight: 1.5 }}>{desc}</div>
            </div>
          ))}
        </div>
      </Card>

      {/* Built-in CoT reasoning dataset */}
      <Card style={{ marginBottom: 16 }}>
        <div style={{ fontWeight: 600, fontSize: 14, color: '#cccccc', marginBottom: 10 }}>
          Chain-of-Thought Reasoning (built-in)
        </div>
        <p style={{ color: theme.textSecondary, fontSize: 13, lineHeight: 1.7, margin: '0 0 12px' }}>
          Loads math and logic pairs where the bot answers with a{' '}
          <code style={{ color: '#93c5fd', background: '#1e3a5f', padding: '1px 5px', borderRadius: 4 }}>
            &lt;redacted_thinking&gt;
          </code>{' '}
          block before the final answer. Requires a <strong style={{ color: theme.text }}>Decoder-Only Transformer</strong>{' '}
          model. Pairs are merged into pre-train data automatically.
        </p>
        <CotSection onRefresh={onRefresh} />
      </Card>

      {/* Status + actions */}
      <Card style={{ marginBottom: 16 }}>
        <div style={{ fontWeight: 600, fontSize: 14, color: theme.text, marginBottom: 14 }}>Built-in Foundation Dataset</div>

        {/* Status row */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 14, padding: '12px 16px', background: theme.panel, borderRadius: 8, border: `1px solid ${loaded ? '#166534' : '#1e1e1e'}`, marginBottom: 16 }}>
          <div style={{ width: 10, height: 10, borderRadius: '50%', background: loaded ? '#22c55e' : '#374151', flexShrink: 0 }} />
          <div style={{ flex: 1 }}>
            <div style={{ fontSize: 13, color: loaded ? '#86efac' : '#94a3b8', fontWeight: 600 }}>
              {loaded ? `${count.toLocaleString()} pre-training pairs loaded` : 'No pre-training data loaded'}
            </div>
            <div style={{ fontSize: 11, color: '#475569', marginTop: 2 }}>
              {loaded
                ? 'Saved to disk. Will persist through restarts.'
                : 'Load the built-in dataset to get started.'}
            </div>
          </div>
          {ptStatus?.persisted && (
            <div style={{ fontSize: 11, color: '#22c55e', background: '#14532d', padding: '2px 8px', borderRadius: 4, flexShrink: 0 }}>
              Saved to disk
            </div>
          )}
        </div>

        <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
          <Btn variant="primary" onClick={handleLoad} disabled={loading}>
            {loading ? 'Loading…' : loaded ? 'Reload Built-in Dataset' : 'Load Built-in Dataset'}
          </Btn>
          {loaded && (
            <Btn onClick={handleDelete} disabled={loading}>
              Delete Pre-train Data
            </Btn>
          )}
        </div>
      </Card>

      {/* What's included */}
      <Card>
        <div style={{ fontWeight: 600, fontSize: 14, color: theme.text, marginBottom: 12 }}>What's Included</div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 8 }}>
          {[
            ['Greetings', '30+ variations of hello, hi, good morning, etc.'],
            ['Farewells', 'Goodbye, bye, see you later, and more.'],
            ['Politeness', 'Thank you, you\'re welcome, sorry, excuse me.'],
            ['Identity Q&A', 'What are you, who made you, are you an AI.'],
            ['Small talk', 'How are you, that\'s interesting, tell me more.'],
            ['Clarification', 'I don\'t understand, can you explain, try again.'],
            ['Acknowledgement', 'Yes, no, okay, got it, makes sense, I see.'],
            ['Requests', 'Can you help, I need help, I have a question.'],
            ['Feedback', 'Great, perfect, not helpful, can you do better.'],
          ].map(([cat, desc]) => (
            <div key={cat} style={{ background: theme.panel, borderRadius: 7, padding: '8px 12px', border: `1px solid ${theme.border}` }}>
              <div style={{ color: theme.text, fontSize: 12, fontWeight: 600, marginBottom: 3 }}>{cat}</div>
              <div style={{ color: '#475569', fontSize: 11, lineHeight: 1.5 }}>{desc}</div>
            </div>
          ))}
        </div>
        <div style={{ marginTop: 14, padding: '10px 14px', background: '#1e3a5f22', border: '1px solid #1e40af', borderRadius: 8, color: '#93c5fd', fontSize: 12 }}>
          On the Training page, enable <strong style={{ color: '#60a5fa' }}>Use Pre-train Data</strong> to prepend these pairs to your training data automatically.
        </div>
      </Card>
    </PageShell>
  )
}

function CotSection({ onRefresh }: { onRefresh: () => void }) {
  const [cot, setCot] = useState<{ loaded: boolean; count: number; decoder_only: boolean | null } | null>(null)
  const [loading, setLoading] = useState(false)
  const [msg, setMsg] = useState<string | null>(null)

  const refresh = async () => {
    try {
      const s = await getCotStatus()
      setCot({ loaded: s.loaded, count: s.count, decoder_only: s.decoder_only })
    } catch {}
  }

  useEffect(() => { refresh() }, [])

  const handleLoad = async () => {
    setLoading(true)
    setMsg(null)
    try {
      const res = await loadCotBuiltin(1) as { count: number; message: string }
      setMsg(res.message)
      await refresh()
      onRefresh()
    } catch (e: unknown) {
      setMsg((e as Error).message)
    } finally {
      setLoading(false)
    }
  }

  const loaded = cot?.loaded ?? false
  const count = cot?.count ?? 0
  const decoderOnly = cot?.decoder_only

  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 14, padding: '12px 16px', background: theme.panel, borderRadius: 8, border: `1px solid ${loaded ? '#166534' : '#1e1e1e'}`, marginBottom: 12 }}>
        <div style={{ width: 10, height: 10, borderRadius: '50%', background: loaded ? '#22c55e' : '#374151', flexShrink: 0 }} />
        <div style={{ flex: 1 }}>
          <div style={{ fontSize: 13, color: loaded ? '#86efac' : '#94a3b8', fontWeight: 600 }}>
            {loaded ? `${count} CoT pairs loaded` : 'No CoT data loaded'}
          </div>
          <div style={{ fontSize: 11, color: '#475569', marginTop: 2 }}>
            {decoderOnly === false
              ? 'Current model is not decoder-only — CoT chat will be unavailable.'
              : loaded
                ? 'Merged into pre-train. Enable reasoning in Inference mode after training.'
                : 'Load built-in math/logic CoT pairs to teach step-by-step reasoning.'}
          </div>
        </div>
      </div>
      {msg && (
        <div style={{ fontSize: 12, color: msg.startsWith('Loaded') || msg.includes('Added') ? '#86efac' : '#fca5a5', marginBottom: 10 }}>
          {msg}
        </div>
      )}
      <Btn variant="primary" onClick={handleLoad} disabled={loading}>
        {loading ? 'Loading…' : loaded ? 'Reload CoT Dataset' : 'Load CoT Dataset'}
      </Btn>
    </div>
  )
}

function Alert({ type, text }: { type: 'success' | 'error' | 'info'; text: string }) {
  const colors = {
    success: { bg: '#14532d22', border: '#166534', text: '#86efac' },
    error:   { bg: '#450a0a',   border: '#7f1d1d', text: '#fca5a5' },
    info:    { bg: '#1e3a5f22', border: '#1e40af', text: '#93c5fd' },
  }
  const c = colors[type]
  return (
    <div style={{ background: c.bg, border: `1px solid ${c.border}`, borderRadius: 8, padding: '10px 14px', marginBottom: 16, color: c.text, fontSize: 13 }}>
      {text}
    </div>
  )
}
