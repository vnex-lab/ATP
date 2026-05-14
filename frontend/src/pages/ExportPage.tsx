import { useState, useEffect } from 'react'
import type { AppStatus, ModelInfo } from '../types'
import { prepareModel, prepareTokenizer, prepareGGUF, generateModelfile, getModelInfo, downloadUrl } from '../api'
import PageShell from '../components/PageShell'
import Card from '../components/Card'
import Btn from '../components/Btn'

export default function ExportPage({ status, onRefresh }: { status: AppStatus; onRefresh: () => void }) {
  const [modelName, setModelName] = useState('vnexai_chatbot')
  const [loading, setLoading] = useState<Record<string, boolean>>({})
  const [ready, setReady] = useState<Record<string, boolean>>({})
  const [msg, setMsg] = useState<{ id: string; type: 'success' | 'error'; text: string } | null>(null)
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null)

  // Ollama
  const [ollamaUser, setOllamaUser] = useState('yourusername')
  const [sysPrompt, setSysPrompt] = useState(`You are ${modelName}, a custom AI assistant trained with VnexAI. Be helpful, friendly, and concise.`)
  const [numCtx, setNumCtx] = useState(512)
  const [tempVal, setTempVal] = useState(0.8)
  const [extraStop, setExtraStop] = useState('')
  const [modelfile, setModelfile] = useState('')

  useEffect(() => {
    if (status.has_model) {
      getModelInfo().then((info) => setModelInfo(info)).catch(() => {})
    }
  }, [status.has_model])

  useEffect(() => {
    setSysPrompt(`You are ${modelName}, a custom AI assistant trained with VnexAI. Be helpful, friendly, and concise.`)
  }, [modelName])

  const setLoad = (id: string, v: boolean) => setLoading((p) => ({ ...p, [id]: v }))
  const setRdy = (id: string, v: boolean) => setReady((p) => ({ ...p, [id]: v }))

  const doExport = async (id: string, fn: () => Promise<unknown>) => {
    setLoad(id, true)
    setMsg(null)
    try {
      await fn()
      setRdy(id, true)
      setMsg({ id, type: 'success', text: 'Ready to download!' })
    } catch (e: unknown) {
      setMsg({ id, type: 'error', text: (e as Error).message })
    } finally {
      setLoad(id, false)
    }
  }

  const handleGenerateModelfile = async () => {
    try {
      const res = await generateModelfile({
        model_name: modelName,
        ollama_username: ollamaUser,
        system_prompt: sysPrompt,
        num_ctx: numCtx,
        temperature: tempVal,
        extra_stop: extraStop,
      }) as { modelfile: string; full_tag: string }
      setModelfile(res.modelfile)
    } catch (e: unknown) {
      setMsg({ id: 'modelfile', type: 'error', text: (e as Error).message })
    }
  }

  const isRnn = modelInfo?.is_rnn ?? false
  const decoderOnly = modelInfo?.decoder_only ?? false
  const canGGUF = !isRnn ? decoderOnly : true

  if (!status.has_model) {
    return (
      <PageShell title="Export" subtitle="Download your trained model.">
        <Card>
          <div style={{ textAlign: 'center', padding: '40px 20px', color: '#475569' }}>
            <div style={{ fontSize: 40, marginBottom: 12 }}>💾</div>
            <div style={{ fontSize: 15, color: '#94a3b8', marginBottom: 6 }}>No model to export</div>
            <div style={{ fontSize: 13 }}>Create and train a model first.</div>
          </div>
        </Card>
      </PageShell>
    )
  }

  return (
    <PageShell title="Export" subtitle="Download your trained model files.">
      {/* Model name */}
      <Card style={{ marginBottom: 16 }}>
        <label style={labelStyle}>Model name (used in all filenames)</label>
        <input
          value={modelName}
          onChange={(e) => setModelName(e.target.value.replace(/\s+/g, '_').toLowerCase())}
          placeholder="vnexai_chatbot"
          style={inputStyle}
        />
        <div style={{ color: '#475569', fontSize: 11, marginTop: 6 }}>
          Files: <code style={{ color: '#94a3b8' }}>{modelName}.bin</code>  ·  <code style={{ color: '#94a3b8' }}>{modelName}.gguf</code>  ·  <code style={{ color: '#94a3b8' }}>{modelName}_tokenizer.bin</code>
        </div>
      </Card>

      {/* Model info */}
      {modelInfo && (
        <Card style={{ marginBottom: 16 }}>
          <div style={{ fontWeight: 600, fontSize: 13, color: '#e2e8f0', marginBottom: 10 }}>Model Information</div>
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
            <Chip label="Type" value={modelInfo.model_type ?? 'Unknown'} />
            <Chip label="Vocab" value={(modelInfo.vocab_size ?? 0).toLocaleString()} />
            {modelInfo.is_trained && modelInfo.final_loss != null && (
              <Chip label="Final Loss" value={modelInfo.final_loss.toFixed(4)} />
            )}
            {!isRnn && modelInfo.num_layers != null && (
              <>
                <Chip label="Layers" value={String(modelInfo.num_layers)} />
                <Chip label="Heads" value={String(modelInfo.num_heads)} />
                <Chip label="Embed Dim" value={String(modelInfo.embedding_dim)} />
              </>
            )}
            {isRnn && (
              <>
                <Chip label="Hidden Dim" value={String(modelInfo.hidden_dim)} />
                <Chip label="Embed Dim" value={String(modelInfo.embedding_dim)} />
              </>
            )}
            <Chip label="Trained" value={modelInfo.is_trained ? 'Yes' : 'No'} color={modelInfo.is_trained ? '#22c55e' : '#475569'} />
          </div>
        </Card>
      )}

      {/* Export cards */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 14, marginBottom: 20 }}>
        {/* .bin Model */}
        <ExportCard
          title=".bin Format"
          icon="📦"
          description="VnexAI native pickle format. Load with the VnexAI Python API."
          badge="Native"
          badgeColor="#3b82f6"
          ready={ready['model']}
          loading={loading['model']}
          msg={msg?.id === 'model' ? msg : null}
          onPrepare={() => doExport('model', prepareModel)}
          onDownload={() => downloadUrl('/export/download-model', modelName)}
          preparelabel="Prepare .bin"
        />

        {/* .gguf */}
        <ExportCard
          title=".gguf Format"
          icon="🦙"
          description={
            !isRnn
              ? decoderOnly
                ? 'Ollama-compatible GGUF. Run with ollama run.'
                : 'Encoder-decoder GGUF. Use Decoder-Only for Ollama.'
              : 'RNN archive format for tooling.'
          }
          badge={!isRnn && decoderOnly ? 'Ollama Ready' : isRnn ? 'Archive' : 'Limited'}
          badgeColor={!isRnn && decoderOnly ? '#22c55e' : '#f59e0b'}
          ready={ready['gguf']}
          loading={loading['gguf']}
          msg={msg?.id === 'gguf' ? msg : null}
          disabled={!isRnn && !decoderOnly}
          disabledReason="Use Decoder-Only Transformer for GGUF export."
          onPrepare={() => doExport('gguf', () => prepareGGUF(modelName))}
          onDownload={() => downloadUrl('/export/download-gguf', modelName)}
          preparelabel="Prepare .gguf"
        />

        {/* Tokenizer */}
        <ExportCard
          title="Tokenizer (.bin)"
          icon="📝"
          description="Vocabulary mapping file. Required to decode model outputs."
          badge="Required for inference"
          badgeColor="#6366f1"
          ready={ready['tok']}
          loading={loading['tok']}
          msg={msg?.id === 'tok' ? msg : null}
          onPrepare={() => doExport('tok', prepareTokenizer)}
          onDownload={() => downloadUrl('/export/download-tokenizer', modelName)}
          preparelabel="Prepare Tokenizer"
        />
      </div>

      {/* Ollama section */}
      {!isRnn && decoderOnly && (
        <Card>
          <div style={{ fontWeight: 600, fontSize: 14, color: '#e2e8f0', marginBottom: 6 }}>🦙 Publish to Ollama Registry</div>
          <p style={{ color: '#94a3b8', fontSize: 13, marginBottom: 16 }}>
            Generate a Modelfile and push your model to ollama.com so anyone can <code style={{ color: '#93c5fd', background: '#1e3a5f', padding: '1px 5px', borderRadius: 4 }}>ollama run</code> it.
          </p>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 14, marginBottom: 14 }}>
            <div>
              <label style={labelStyle}>Ollama username</label>
              <input value={ollamaUser} onChange={(e) => setOllamaUser(e.target.value)} style={inputStyle} />
            </div>
            <div>
              <label style={labelStyle}>Context length (num_ctx)</label>
              <input type="number" value={numCtx} onChange={(e) => setNumCtx(Number(e.target.value))} min={64} max={4096} step={64} style={inputStyle} />
            </div>
            <div>
              <label style={labelStyle}>Default temperature</label>
              <input type="range" min={0.1} max={2.0} step={0.1} value={tempVal} onChange={(e) => setTempVal(Number(e.target.value))} style={{ width: '100%', accentColor: '#3b82f6' }} />
              <div style={{ color: '#475569', fontSize: 11, marginTop: 4 }}>{tempVal.toFixed(1)}</div>
            </div>
            <div>
              <label style={labelStyle}>Extra stop token (optional)</label>
              <input value={extraStop} onChange={(e) => setExtraStop(e.target.value)} placeholder="e.g. </s>" style={inputStyle} />
            </div>
          </div>
          <div style={{ marginBottom: 14 }}>
            <label style={labelStyle}>System prompt</label>
            <textarea value={sysPrompt} onChange={(e) => setSysPrompt(e.target.value)} rows={2} style={{ ...inputStyle, resize: 'vertical' }} />
          </div>

          <Btn onClick={handleGenerateModelfile}>Generate Modelfile</Btn>

          {modelfile && (
            <div style={{ marginTop: 14 }}>
              <label style={labelStyle}>Generated Modelfile</label>
              <pre style={{
                background: '#0a0a0a', border: '1px solid #1e1e1e', borderRadius: 8, padding: '12px 14px',
                color: '#93c5fd', fontSize: 12, fontFamily: 'monospace', overflowX: 'auto', whiteSpace: 'pre-wrap',
              }}>{modelfile}</pre>
              <button
                onClick={() => {
                  const a = document.createElement('a')
                  a.href = 'data:text/plain;charset=utf-8,' + encodeURIComponent(modelfile)
                  a.download = 'Modelfile'
                  a.click()
                }}
                style={{ marginTop: 10, ...btnStyle('#1c1c1c', '#94a3b8') }}
              >📥 Download Modelfile</button>

              <div style={{ marginTop: 20 }}>
                <div style={{ fontWeight: 600, fontSize: 13, color: '#e2e8f0', marginBottom: 12 }}>🚀 Push to Ollama Registry</div>
                {[
                  ['1. Install Ollama', 'curl -fsSL https://ollama.com/install.sh | sh'],
                  ['2. Create model locally', `ollama create ${ollamaUser}/${modelName} -f Modelfile`],
                  ['3. Test it', `ollama run ${ollamaUser}/${modelName}`],
                  ['4. Login', 'ollama login'],
                  ['5. Push to registry', `ollama push ${ollamaUser}/${modelName}`],
                ].map(([label, cmd]) => (
                  <div key={label} style={{ marginBottom: 10 }}>
                    <div style={{ fontSize: 12, color: '#94a3b8', marginBottom: 4 }}>{label}</div>
                    <pre style={{
                      background: '#0a0a0a', border: '1px solid #1e1e1e', borderRadius: 7,
                      padding: '8px 12px', color: '#e2e8f0', fontSize: 12, fontFamily: 'monospace', margin: 0,
                    }}>{cmd}</pre>
                  </div>
                ))}
                <div style={{ marginTop: 10, padding: '10px 14px', background: '#1e3a5f22', border: '1px solid #1e40af', borderRadius: 8, color: '#93c5fd', fontSize: 12 }}>
                  Once pushed: <strong>https://ollama.com/{ollamaUser}/{modelName}</strong>
                </div>
              </div>
            </div>
          )}
        </Card>
      )}
    </PageShell>
  )
}

function ExportCard({
  title, icon, description, badge, badgeColor, ready, loading, msg, disabled, disabledReason,
  onPrepare, onDownload, preparelabel,
}: {
  title: string; icon: string; description: string; badge: string; badgeColor: string;
  ready: boolean; loading: boolean; msg: { type: 'success' | 'error'; text: string } | null;
  disabled?: boolean; disabledReason?: string;
  onPrepare: () => void; onDownload: () => void; preparelabel: string;
}) {
  return (
    <div style={{
      background: '#0f0f0f', border: '1px solid #1e1e1e', borderRadius: 10, padding: '16px',
      display: 'flex', flexDirection: 'column', gap: 10, opacity: disabled ? 0.6 : 1,
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <span style={{ fontSize: 20 }}>{icon}</span>
        <span style={{ fontWeight: 600, fontSize: 13, color: '#e2e8f0' }}>{title}</span>
      </div>
      <span style={{ fontSize: 11, color: badgeColor, background: badgeColor + '22', padding: '2px 8px', borderRadius: 20, alignSelf: 'flex-start' }}>
        {badge}
      </span>
      <p style={{ fontSize: 12, color: '#94a3b8', lineHeight: 1.5 }}>{description}</p>
      {disabled && disabledReason && (
        <div style={{ fontSize: 11, color: '#f59e0b' }}>{disabledReason}</div>
      )}
      {msg && (
        <div style={{ fontSize: 12, color: msg.type === 'success' ? '#86efac' : '#fca5a5' }}>{msg.text}</div>
      )}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 8, marginTop: 'auto' }}>
        <button
          onClick={onPrepare}
          disabled={disabled || loading}
          style={btnStyle(disabled || loading ? '#1c1c1c' : '#1c1c1c', disabled || loading ? '#374151' : '#94a3b8')}
        >
          {loading ? 'Preparing…' : preparelabel}
        </button>
        {ready && (
          <button onClick={onDownload} style={btnStyle('#2563eb', '#fff')}>
            📥 Download
          </button>
        )}
      </div>
    </div>
  )
}

function Chip({ label, value, color }: { label: string; value: string; color?: string }) {
  return (
    <div style={{ background: '#111', borderRadius: 6, padding: '5px 10px', border: '1px solid #1e1e1e' }}>
      <span style={{ fontSize: 11, color: '#475569' }}>{label}: </span>
      <span style={{ fontSize: 12, color: color ?? '#e2e8f0', fontWeight: 500 }}>{value}</span>
    </div>
  )
}

function btnStyle(bg: string, color: string): React.CSSProperties {
  return {
    width: '100%', padding: '8px 12px', borderRadius: 7, border: '1px solid #252525',
    background: bg, color, fontSize: 12, cursor: 'pointer', fontFamily: 'inherit', fontWeight: 500,
  }
}

const labelStyle: React.CSSProperties = {
  fontSize: 12, color: '#94a3b8', display: 'block', marginBottom: 6, fontWeight: 500,
}

const inputStyle: React.CSSProperties = {
  width: '100%', background: '#111', border: '1px solid #252525', borderRadius: 7,
  padding: '7px 10px', color: '#e2e8f0', fontSize: 13, outline: 'none', fontFamily: 'inherit',
}
