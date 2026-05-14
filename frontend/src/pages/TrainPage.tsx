import { useState, useEffect, useRef } from 'react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'
import type { AppStatus, TrainingState } from '../types'
import { startTraining, subscribeToTraining } from '../api'
import PageShell from '../components/PageShell'
import Card from '../components/Card'
import Btn from '../components/Btn'
import Badge from '../components/Badge'

export default function TrainPage({ status, onRefresh }: { status: AppStatus; onRefresh: () => void }) {
  const [epochs, setEpochs] = useState(100)
  const [batchSize, setBatchSize] = useState(16)
  const [shuffle, setShuffle] = useState(true)
  const [useSft, setUseSft] = useState(false)
  const [loading, setLoading] = useState(false)
  const [msg, setMsg] = useState<{ type: 'success' | 'error' | 'info'; text: string } | null>(null)
  const [trainingData, setTrainingData] = useState<TrainingState>(status.training)
  const unsubRef = useRef<(() => void) | null>(null)

  const isDecoderOnly = (status.model_type === 'TransformerDecoder') ||
    (status.model_config as Record<string, unknown> | null)?.model_type === 'transformer_decoder'

  useEffect(() => {
    setTrainingData(status.training)
  }, [status.training])

  useEffect(() => {
    return () => { unsubRef.current?.() }
  }, [])

  const handleStart = async () => {
    if (!status.has_model) return setMsg({ type: 'error', text: 'Create a model first.' })
    if (!status.has_training_data) return setMsg({ type: 'error', text: 'Load training data first.' })
    if (!status.has_tokenizer) return setMsg({ type: 'error', text: 'Build vocabulary first.' })
    setMsg(null)
    setLoading(true)
    try {
      await startTraining({ epochs, batch_size: batchSize, shuffle_data: shuffle, use_sft: useSft })
      const unsub = subscribeToTraining(
        (data) => setTrainingData(data as TrainingState),
        () => { setLoading(false); onRefresh() }
      )
      unsubRef.current = unsub
      setMsg({ type: 'info', text: 'Training started…' })
    } catch (e: unknown) {
      setMsg({ type: 'error', text: (e as Error).message })
      setLoading(false)
    }
  }

  const td = trainingData
  const isTraining = td.is_training
  const isDone = td.status === 'done'
  const isError = td.status === 'error'

  const chartData = td.losses.map((loss, i) => ({ epoch: i + 1, loss: Number(loss.toFixed(4)) }))

  return (
    <PageShell
      title="Training"
      subtitle="Train your model on the uploaded conversation data."
      badge={
        isDone ? <Badge color="green">Training Complete</Badge> :
        isTraining ? <Badge color="blue">Training…</Badge> :
        status.is_trained ? <Badge color="green">Trained</Badge> : undefined
      }
    >
      {msg && <Alert type={msg.type} text={msg.text} />}
      {isError && td.error && <Alert type="error" text={`Training failed: ${td.error}`} />}

      {!status.has_model && <Alert type="warn" text="Create a model on the Model page first." />}

      {/* Config */}
      <Card style={{ marginBottom: 16 }}>
        <div style={{ fontWeight: 600, fontSize: 14, color: '#e2e8f0', marginBottom: 14 }}>Training Configuration</div>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 14 }}>
          <Field label="Epochs">
            <NumInput value={epochs} onChange={setEpochs} min={1} max={10000} step={5} disabled={isTraining} />
          </Field>
          <Field label="Batch Size" note={td.gpu_available ? 'GPU active' : 'CPU mode'}>
            <NumInput value={batchSize} onChange={setBatchSize} min={1} max={512} step={1} disabled={isTraining} />
          </Field>
          <Field label="Options">
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8, paddingTop: 4 }}>
              <label style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer' }}>
                <input type="checkbox" checked={shuffle} onChange={(e) => setShuffle(e.target.checked)} disabled={isTraining}
                  style={{ accentColor: '#3b82f6', width: 14, height: 14 }} />
                <span style={{ fontSize: 13, color: '#94a3b8' }}>Shuffle data</span>
              </label>
              {isDecoderOnly && (
                <label style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer' }}>
                  <input type="checkbox" checked={useSft} onChange={(e) => setUseSft(e.target.checked)} disabled={isTraining}
                    style={{ accentColor: '#3b82f6', width: 14, height: 14 }} />
                  <span style={{ fontSize: 13, color: '#94a3b8' }}>SFT Loss <span style={{ color: '#475569', fontSize: 11 }}>(decoder-only)</span></span>
                </label>
              )}
            </div>
          </Field>
        </div>

        {/* Tips */}
        <div style={{ background: '#111', border: '1px solid #1e1e1e', borderRadius: 8, padding: '10px 14px', marginTop: 14 }}>
          <div style={{ fontSize: 11, color: '#475569', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 6 }}>Quick Tips</div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
            {[
              'Loss stuck? Try a higher LR (0.005–0.01) in Model Setup.',
              'Loss bouncing? Lower LR or increase Gradient Clip.',
              'Word salad after 100 epochs? You need more training data.',
              'GPU active = much faster training per epoch.',
            ].map((t) => (
              <div key={t} style={{ fontSize: 12, color: '#475569' }}>· {t}</div>
            ))}
          </div>
        </div>

        <div style={{ marginTop: 14 }}>
          <Btn
            variant="primary"
            onClick={handleStart}
            disabled={isTraining || !status.has_model || loading}
          >
            {isTraining ? `Training… Epoch ${td.current_epoch}/${td.total_epochs}` : 'Start Training'}
          </Btn>
        </div>
      </Card>

      {/* Progress */}
      {(isTraining || isDone || td.losses.length > 0) && (
        <Card style={{ marginBottom: 16 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 14 }}>
            <div style={{ fontWeight: 600, fontSize: 14, color: '#e2e8f0' }}>
              {isDone ? 'Training Complete' : isTraining ? 'Training Progress' : 'Last Training Run'}
            </div>
            <div style={{ display: 'flex', gap: 10 }}>
              {td.gpu_available && (
                <StatChip label="Mode" value="🚀 GPU" />
              )}
              {td.current_epoch > 0 && (
                <StatChip label="Loss" value={td.avg_loss.toFixed(4)} />
              )}
            </div>
          </div>

          {/* Progress bar */}
          {(isTraining || isDone) && (
            <div style={{ marginBottom: 16 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, color: '#94a3b8', marginBottom: 6 }}>
                <span>Epoch {td.current_epoch} of {td.total_epochs}</span>
                <span>{(td.progress * 100).toFixed(1)}%</span>
              </div>
              <div style={{ height: 6, background: '#1c1c1c', borderRadius: 3, overflow: 'hidden' }}>
                <div style={{
                  height: '100%',
                  width: `${td.progress * 100}%`,
                  background: isDone ? '#22c55e' : 'linear-gradient(90deg, #2563eb, #3b82f6)',
                  borderRadius: 3,
                  transition: 'width 0.4s ease',
                }} />
              </div>
              {isTraining && td.total_batches > 0 && (
                <div style={{ fontSize: 11, color: '#475569', marginTop: 6 }}>
                  Batch {td.current_batch}/{td.total_batches} · Current loss: {td.current_loss.toFixed(4)}
                </div>
              )}
            </div>
          )}

          {/* Chart */}
          {chartData.length > 1 && (
            <div style={{ height: 220 }}>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartData} margin={{ top: 5, right: 10, left: -10, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e1e1e" />
                  <XAxis
                    dataKey="epoch"
                    stroke="#374151"
                    tick={{ fill: '#475569', fontSize: 11 }}
                    label={{ value: 'Epoch', position: 'insideBottom', offset: -2, fill: '#475569', fontSize: 11 }}
                  />
                  <YAxis
                    stroke="#374151"
                    tick={{ fill: '#475569', fontSize: 11 }}
                    label={{ value: 'Loss', angle: -90, position: 'insideLeft', fill: '#475569', fontSize: 11 }}
                  />
                  <Tooltip
                    contentStyle={{ background: '#1c1c1c', border: '1px solid #252525', borderRadius: 7, fontSize: 12 }}
                    labelStyle={{ color: '#94a3b8' }}
                    itemStyle={{ color: '#3b82f6' }}
                  />
                  <Line
                    type="monotone"
                    dataKey="loss"
                    stroke="#3b82f6"
                    strokeWidth={2}
                    dot={chartData.length < 30 ? { fill: '#3b82f6', r: 3 } : false}
                    activeDot={{ r: 5, fill: '#60a5fa' }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}

          {isDone && td.losses.length > 0 && (
            <div style={{ marginTop: 12, display: 'flex', gap: 10, flexWrap: 'wrap' }}>
              <StatChip label="Start Loss" value={td.losses[0].toFixed(4)} />
              <StatChip label="Final Loss" value={td.losses[td.losses.length - 1].toFixed(4)} />
              <StatChip label="Improvement"
                value={`${(((td.losses[0] - td.losses[td.losses.length - 1]) / td.losses[0]) * 100).toFixed(1)}%`}
              />
              <StatChip label="Epochs" value={td.total_epochs.toString()} />
            </div>
          )}
        </Card>
      )}

      {/* Dataset info */}
      {status.has_training_data && (
        <Card>
          <div style={{ fontWeight: 600, fontSize: 13, color: '#e2e8f0', marginBottom: 10 }}>Dataset</div>
          <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
            <StatChip label="Pairs" value={status.training_data_count.toLocaleString()} />
            {status.training_data_profile && <StatChip label="Profile" value={status.training_data_profile} />}
            {status.has_tokenizer && <StatChip label="Vocab" value={`${status.tokenizer_vocab_size.toLocaleString()} tokens`} />}
          </div>
          {status.training_data_count < 500 && (
            <div style={{ marginTop: 10, color: '#f59e0b', fontSize: 12 }}>
              ⚠ Small dataset ({status.training_data_count} pairs). Aim for 1,000+ for better results.
            </div>
          )}
        </Card>
      )}
    </PageShell>
  )
}

function Field({ label, note, children }: { label: string; note?: string; children: React.ReactNode }) {
  return (
    <div>
      <label style={{ fontSize: 12, color: '#94a3b8', display: 'block', marginBottom: 6, fontWeight: 500 }}>
        {label}{note && <span style={{ color: '#374151', marginLeft: 6, fontWeight: 400 }}>{note}</span>}
      </label>
      {children}
    </div>
  )
}

function NumInput({ value, onChange, min, max, step, disabled }: {
  value: number; onChange: (v: number) => void; min: number; max: number; step: number; disabled?: boolean
}) {
  return (
    <input type="number" value={value} onChange={(e) => onChange(Number(e.target.value))}
      min={min} max={max} step={step} disabled={disabled}
      style={{ width: '100%', background: '#111', border: '1px solid #252525', borderRadius: 7, padding: '7px 10px', color: '#e2e8f0', fontSize: 13, outline: 'none', fontFamily: 'inherit', opacity: disabled ? 0.5 : 1 }}
    />
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
