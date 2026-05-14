import { useState } from 'react'
import type { AppStatus } from '../types'
import { estimateVocab, buildVocab, createModel } from '../api'
import PageShell from '../components/PageShell'
import Card from '../components/Card'
import Btn from '../components/Btn'
import Badge from '../components/Badge'

type ModelType = 'transformer_decoder' | 'transformer_enc_dec' | 'rnn'

export default function ModelPage({ status, onRefresh }: { status: AppStatus; onRefresh: () => void }) {
  const [loading, setLoading] = useState(false)
  const [msg, setMsg] = useState<{ type: 'success' | 'error' | 'info' | 'warn'; text: string } | null>(null)

  // Tokenizer
  const [maxVocab, setMaxVocab] = useState(12000)
  const [padVocab, setPadVocab] = useState(true)
  const [padTarget, setPadTarget] = useState(6000)
  const [vocabResult, setVocabResult] = useState<{ vocab_size: number } | null>(null)

  // Model architecture
  const [modelType, setModelType] = useState<ModelType>('transformer_decoder')
  const [embDim, setEmbDim] = useState(256)
  const [numHeads, setNumHeads] = useState(8)
  const [numLayers, setNumLayers] = useState(4)
  const [ffDim, setFfDim] = useState(1024)
  const [maxLen, setMaxLen] = useState(128)
  const [lr, setLr] = useState(0.003)
  const [hiddenDim, setHiddenDim] = useState(256)
  const [rnnMaxLen, setRnnMaxLen] = useState(50)
  const [rnnLr, setRnnLr] = useState(0.05)

  // Advanced
  const [optimizer, setOptimizer] = useState('adam')
  const [scheduler, setScheduler] = useState('warmup_cosine')
  const [dropout, setDropout] = useState(0.1)
  const [weightDecay, setWeightDecay] = useState(0.01)
  const [warmupEpochs, setWarmupEpochs] = useState(5)
  const [gradClip, setGradClip] = useState(5.0)
  const [showAdvanced, setShowAdvanced] = useState(false)

  const [modelResult, setModelResult] = useState<{
    total_params: number; total_params_m: number; gpu_available: boolean; gpu_info: string; model_type: string; vocab_size: number
  } | null>(null)

  const ok = (text: string) => setMsg({ type: 'success', text })
  const err = (text: string) => setMsg({ type: 'error', text })
  const warn = (text: string) => setMsg({ type: 'warn', text })

  const handleEstimate = async () => {
    setLoading(true)
    try {
      const res = await estimateVocab(maxVocab) as { unique_words: number }
      setMsg({ type: 'info', text: `~${res.unique_words.toLocaleString()} unique word tokens found in training data.` })
    } catch (e: unknown) { err((e as Error).message) } finally { setLoading(false) }
  }

  const handleBuildVocab = async () => {
    setLoading(true)
    setMsg(null)
    try {
      const res = await buildVocab({ max_vocab_size: maxVocab, pad_vocab: padVocab, pad_target: padTarget }) as { vocab_size: number }
      setVocabResult(res)
      ok(`Vocabulary built! ${res.vocab_size.toLocaleString()} tokens.`)
      onRefresh()
    } catch (e: unknown) { err((e as Error).message) } finally { setLoading(false) }
  }

  const handleCreateModel = async () => {
    if (!status.has_tokenizer && !vocabResult) return err('Build vocabulary first.')
    setLoading(true)
    setMsg(null)
    try {
      const config: Record<string, unknown> = {
        model_type: modelType,
        embedding_dim: embDim,
        max_length: modelType === 'rnn' ? rnnMaxLen : maxLen,
        learning_rate: modelType === 'rnn' ? rnnLr : lr,
      }
      if (modelType === 'rnn') {
        config.hidden_dim = hiddenDim
      } else {
        config.num_heads = numHeads
        config.num_layers = numLayers
        config.ff_dim = ffDim
        config.optimizer = optimizer
        config.scheduler = scheduler
        config.dropout_rate = dropout
        config.weight_decay = weightDecay
        config.warmup_epochs = warmupEpochs
        config.grad_clip = gradClip
      }
      const res = await createModel(config) as typeof modelResult
      setModelResult(res)
      ok(`Model created! ${res!.total_params.toLocaleString()} parameters (${res!.total_params_m}M). ${res!.gpu_available ? 'GPU ready.' : 'CPU mode.'}`)
      onRefresh()
    } catch (e: unknown) { err((e as Error).message) } finally { setLoading(false) }
  }

  const isTransformer = modelType !== 'rnn'
  const headError = isTransformer && embDim % numHeads !== 0
    ? `${embDim} is not divisible by ${numHeads} heads` : null

  const approxParams = isTransformer
    ? (() => {
        const vs = status.tokenizer_vocab_size || 6000
        const attn = 4 * embDim * embDim
        const ff = embDim * ffDim + ffDim * embDim
        const enc = modelType === 'transformer_enc_dec' ? numLayers * (attn + ff) : 0
        const dec = numLayers * ((modelType === 'transformer_enc_dec' ? 2 : 1) * attn + ff)
        return vs * embDim + enc + dec + embDim * vs + vs
      })()
    : (() => {
        const vs = status.tokenizer_vocab_size || 6000
        return vs * embDim + embDim * hiddenDim * 2 + hiddenDim * hiddenDim * 2 + hiddenDim * 4 + hiddenDim * vs + vs
      })()
  const vramMb = (approxParams * 4) / (1024 * 1024)

  return (
    <PageShell
      title="Model Setup"
      subtitle="Configure your tokenizer and neural network architecture."
      badge={status.has_model ? <Badge color="blue">{status.model_type ?? 'Model ready'}</Badge> : undefined}
    >
      {msg && <Alert type={msg.type} text={msg.text} />}

      {!status.has_training_data && (
        <Alert type="warn" text="Load training data first before building the vocabulary." />
      )}

      {/* Step 1: Tokenizer */}
      <Card style={{ marginBottom: 16 }}>
        <StepLabel step={1} done={status.has_tokenizer}>Tokenizer / Vocabulary</StepLabel>
        <p style={{ color: '#94a3b8', fontSize: 13, margin: '6px 0 16px' }}>
          Builds a word-to-index mapping from your training text. Vocabulary size = number of unique words found (capped by maximum).
        </p>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 14 }}>
          <div>
            <label style={labelStyle}>Maximum vocabulary size</label>
            <NumInput value={maxVocab} onChange={setMaxVocab} min={1000} max={50000} step={500} />
          </div>
          <div>
            <label style={labelStyle}>Target size after padding</label>
            <NumInput value={padTarget} onChange={setPadTarget} min={2000} max={50000} step={500} disabled={!padVocab} />
          </div>
        </div>
        <label style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer', marginBottom: 14 }}>
          <input
            type="checkbox"
            checked={padVocab}
            onChange={(e) => setPadVocab(e.target.checked)}
            style={{ accentColor: '#3b82f6', width: 14, height: 14 }}
          />
          <span style={{ fontSize: 13, color: '#94a3b8' }}>
            Pad with common English words <span style={{ color: '#475569' }}>(recommended for small datasets)</span>
          </span>
        </label>
        <div style={{ display: 'flex', gap: 10 }}>
          <Btn onClick={handleEstimate} disabled={!status.has_training_data || loading}>Estimate Unique Words</Btn>
          <Btn variant="primary" onClick={handleBuildVocab} disabled={!status.has_training_data || loading}>
            {loading ? 'Building…' : 'Build Vocabulary'}
          </Btn>
        </div>
        {(status.has_tokenizer || vocabResult) && (
          <div style={{ marginTop: 12, display: 'flex', gap: 10 }}>
            <StatChip label="Vocab Size" value={(vocabResult?.vocab_size ?? status.tokenizer_vocab_size).toLocaleString()} />
          </div>
        )}
      </Card>

      {/* Step 2: Architecture */}
      <Card style={{ marginBottom: 16 }}>
        <StepLabel step={2} done={status.has_model}>Model Architecture</StepLabel>

        {/* Architecture selector */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8, margin: '12px 0 16px' }}>
          {([
            ['transformer_decoder', 'Transformer Decoder-Only', 'Best choice. Ollama-ready GGUF export. Like GPT.'],
            ['transformer_enc_dec', 'Transformer Encoder-Decoder', 'Legacy. Standard seq2seq Transformer.'],
            ['rnn', 'RNN (Legacy)', 'Simple recurrent network. Less capable.'],
          ] as const).map(([id, label, desc]) => (
            <label
              key={id}
              style={{
                display: 'flex',
                alignItems: 'flex-start',
                gap: 10,
                padding: '10px 14px',
                borderRadius: 8,
                border: `1px solid ${modelType === id ? '#3b82f6' : '#1e1e1e'}`,
                background: modelType === id ? '#1e3a5f22' : '#0f0f0f',
                cursor: 'pointer',
                transition: 'all 0.1s',
              }}
            >
              <input
                type="radio"
                name="arch"
                value={id}
                checked={modelType === id}
                onChange={() => setModelType(id)}
                style={{ accentColor: '#3b82f6', marginTop: 2 }}
              />
              <div>
                <div style={{ color: '#e2e8f0', fontSize: 13, fontWeight: 500 }}>{label}</div>
                <div style={{ color: '#475569', fontSize: 12, marginTop: 2 }}>{desc}</div>
              </div>
            </label>
          ))}
        </div>

        {/* GPU Guide */}
        <div style={{ background: '#111', border: '1px solid #1e1e1e', borderRadius: 8, padding: '10px 14px', marginBottom: 16 }}>
          <div style={{ fontSize: 11, color: '#475569', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 8 }}>GPU Size Guide</div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 4 }}>
            {[
              ['CPU / No GPU', 'embed=128, heads=4, layers=2, ff=512'],
              ['GTX 1650 4GB', 'embed=256, heads=8, layers=4, ff=1024'],
              ['RTX 3060 12GB', 'embed=512, heads=8, layers=6, ff=2048'],
              ['RTX 4090 24GB', 'embed=1024, heads=16, layers=12, ff=4096'],
            ].map(([gpu, params]) => (
              <div key={gpu} style={{ fontSize: 11 }}>
                <span style={{ color: '#94a3b8' }}>{gpu}: </span>
                <span style={{ color: '#64748b', fontFamily: 'monospace' }}>{params}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Params */}
        {isTransformer ? (
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 12 }}>
            <Field label="Embedding Dim" note="Divisible by heads">
              <NumInput value={embDim} onChange={setEmbDim} min={32} max={16384} step={32} />
            </Field>
            <Field label="Attention Heads">
              <NumInput value={numHeads} onChange={setNumHeads} min={1} max={32} step={1} />
              {headError && <div style={{ color: '#f87171', fontSize: 11, marginTop: 4 }}>{headError}</div>}
            </Field>
            <Field label="Layers">
              <NumInput value={numLayers} onChange={setNumLayers} min={1} max={24} step={1} />
            </Field>
            <Field label="FF Dim" note="~4× embed dim">
              <NumInput value={ffDim} onChange={setFfDim} min={128} max={65536} step={128} />
            </Field>
            <Field label="Max Sequence Length">
              <NumInput value={maxLen} onChange={setMaxLen} min={10} max={500} step={10} />
            </Field>
            <Field label="Learning Rate">
              <input
                type="number"
                value={lr}
                onChange={(e) => setLr(Number(e.target.value))}
                step={0.0001}
                min={0.000001}
                max={0.1}
                style={inputStyle}
              />
            </Field>
          </div>
        ) : (
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 12 }}>
            <Field label="Embedding Dim">
              <NumInput value={embDim} onChange={setEmbDim} min={32} max={16384} step={32} />
            </Field>
            <Field label="Hidden Dim">
              <NumInput value={hiddenDim} onChange={setHiddenDim} min={64} max={32768} step={64} />
            </Field>
            <Field label="Max Sequence Length">
              <NumInput value={rnnMaxLen} onChange={setRnnMaxLen} min={10} max={500} step={10} />
            </Field>
            <Field label="Learning Rate">
              <input
                type="number"
                value={rnnLr}
                onChange={(e) => setRnnLr(Number(e.target.value))}
                step={0.001}
                min={0.001}
                max={1.0}
                style={inputStyle}
              />
            </Field>
          </div>
        )}

        {/* Advanced (Transformer only) */}
        {isTransformer && (
          <div style={{ marginTop: 14 }}>
            <button
              onClick={() => setShowAdvanced(!showAdvanced)}
              style={{ background: 'none', border: 'none', color: '#94a3b8', cursor: 'pointer', fontSize: 12, display: 'flex', alignItems: 'center', gap: 6 }}
            >
              <span style={{ transform: showAdvanced ? 'rotate(90deg)' : 'none', display: 'inline-block', transition: 'transform 0.15s' }}>▶</span>
              Advanced Training Parameters
            </button>
            {showAdvanced && (
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 12, marginTop: 12 }}>
                <Field label="Optimizer">
                  <Select value={optimizer} onChange={setOptimizer} options={[['adam','Adam'],['adamw','AdamW'],['sgd','SGD']]} />
                </Field>
                <Field label="LR Scheduler">
                  <Select value={scheduler} onChange={setScheduler} options={[
                    ['warmup_cosine','Warmup + Cosine'],['cosine','Cosine'],['linear','Linear'],
                    ['warmup_linear','Warmup + Linear'],['constant','Constant'],
                  ]} />
                </Field>
                <Field label="Dropout Rate">
                  <NumInput value={dropout} onChange={setDropout} min={0} max={0.5} step={0.05} />
                </Field>
                <Field label="Weight Decay">
                  <NumInput value={weightDecay} onChange={setWeightDecay} min={0} max={1} step={0.001} />
                </Field>
                <Field label="Warmup Epochs">
                  <NumInput value={warmupEpochs} onChange={setWarmupEpochs} min={0} max={50} step={1} />
                </Field>
                <Field label="Gradient Clip">
                  <NumInput value={gradClip} onChange={setGradClip} min={0.1} max={50} step={0.5} />
                </Field>
              </div>
            )}
          </div>
        )}

        {/* Parameter estimate */}
        <div style={{ display: 'flex', gap: 10, marginTop: 16, flexWrap: 'wrap' }}>
          <StatChip label="Est. Parameters" value={`${approxParams.toLocaleString()} (${(approxParams / 1e6).toFixed(1)}M)`} />
          <StatChip label="Est. VRAM" value={`~${vramMb.toFixed(0)} MB`} />
        </div>
        {approxParams > 1e9 && (
          <div style={{ marginTop: 8, color: '#f87171', fontSize: 12 }}>{(approxParams / 1e9).toFixed(1)}B params — needs high-end GPU</div>
        )}
        {approxParams > 1e8 && approxParams <= 1e9 && (
          <div style={{ marginTop: 8, color: '#f59e0b', fontSize: 12 }}>{(approxParams / 1e6).toFixed(0)}M params — large model, ensure enough VRAM</div>
        )}

        <div style={{ marginTop: 16 }}>
          <Btn
            variant="primary"
            onClick={handleCreateModel}
            disabled={!status.has_tokenizer || loading || !!headError}
          >
            {loading ? 'Creating…' : 'Create Model'}
          </Btn>
        </div>

        {modelResult && (
          <div style={{ marginTop: 14, padding: '12px 14px', background: '#14532d22', border: '1px solid #166534', borderRadius: 8 }}>
            <div style={{ color: '#86efac', fontWeight: 600, marginBottom: 8 }}>Model Created</div>
            <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
              <StatChip label="Parameters" value={`${modelResult.total_params.toLocaleString()} (${modelResult.total_params_m}M)`} />
              <StatChip label="Compute" value={modelResult.gpu_available ? 'GPU (CuPy)' : 'CPU (NumPy)'} />
              <StatChip label="Type" value={modelResult.model_type} />
              <StatChip label="Vocab" value={modelResult.vocab_size.toLocaleString()} />
            </div>
          </div>
        )}
      </Card>
    </PageShell>
  )
}

function StepLabel({ step, done, children }: { step: number; done: boolean; children: React.ReactNode }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
      <span style={{
        width: 22, height: 22, borderRadius: 6,
        background: done ? '#14532d' : '#1c1c1c',
        border: `1px solid ${done ? '#16a34a' : '#252525'}`,
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        fontSize: 11, color: done ? '#22c55e' : '#64748b', fontFamily: 'monospace', flexShrink: 0,
      }}>{done ? '✓' : step}</span>
      <span style={{ fontWeight: 600, fontSize: 14, color: '#e2e8f0' }}>{children}</span>
    </div>
  )
}

function Field({ label, note, children }: { label: string; note?: string; children: React.ReactNode }) {
  return (
    <div>
      <label style={{ ...labelStyle, display: 'flex', justifyContent: 'space-between' }}>
        <span>{label}</span>
        {note && <span style={{ color: '#374151', fontWeight: 400 }}>{note}</span>}
      </label>
      {children}
    </div>
  )
}

function NumInput({ value, onChange, min, max, step, disabled }: {
  value: number; onChange: (v: number) => void; min: number; max: number; step: number; disabled?: boolean
}) {
  return (
    <input
      type="number"
      value={value}
      onChange={(e) => onChange(Number(e.target.value))}
      min={min} max={max} step={step}
      disabled={disabled}
      style={{ ...inputStyle, opacity: disabled ? 0.5 : 1 }}
    />
  )
}

function Select({ value, onChange, options }: { value: string; onChange: (v: string) => void; options: [string, string][] }) {
  return (
    <select value={value} onChange={(e) => onChange(e.target.value)} style={inputStyle}>
      {options.map(([v, l]) => <option key={v} value={v}>{l}</option>)}
    </select>
  )
}

function StatChip({ label, value }: { label: string; value: string }) {
  return (
    <div style={{ background: '#111', borderRadius: 6, padding: '5px 10px', border: '1px solid #1e1e1e' }}>
      <span style={{ fontSize: 11, color: '#475569' }}>{label}: </span>
      <span style={{ fontSize: 12, color: '#e2e8f0', fontWeight: 500 }}>{value}</span>
    </div>
  )
}

function Alert({ type, text }: { type: 'success' | 'error' | 'info' | 'warn'; text: string }) {
  const colors = {
    success: { bg: '#14532d22', border: '#166534', text: '#86efac' },
    error: { bg: '#450a0a', border: '#7f1d1d', text: '#fca5a5' },
    info: { bg: '#1e3a5f22', border: '#1e40af', text: '#93c5fd' },
    warn: { bg: '#451a0322', border: '#92400e', text: '#fcd34d' },
  }
  const c = colors[type]
  return (
    <div style={{ background: c.bg, border: `1px solid ${c.border}`, borderRadius: 8, padding: '10px 14px', marginBottom: 16, color: c.text, fontSize: 13 }}>
      {text}
    </div>
  )
}

const labelStyle: React.CSSProperties = {
  fontSize: 12, color: '#94a3b8', display: 'block', marginBottom: 6, fontWeight: 500,
}

const inputStyle: React.CSSProperties = {
  width: '100%', background: '#111', border: '1px solid #252525', borderRadius: 7,
  padding: '7px 10px', color: '#e2e8f0', fontSize: 13, outline: 'none', fontFamily: 'inherit',
}
