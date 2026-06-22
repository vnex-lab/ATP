import { useState, useEffect } from 'react'
import type { AppStatus, ModelInfo } from '../types'
import { prepareExportBundle, getModelInfo, downloadUrl } from '../api'
import PageShell from '../components/PageShell'
import Card from '../components/Card'
import Btn from '../components/Btn'
import { theme } from '../theme'

export default function ExportPage({ status }: { status: AppStatus; onRefresh: () => void }) {
  const [modelName, setModelName] = useState('vnexai_model')
  const [loading, setLoading] = useState(false)
  const [ready, setReady] = useState(false)
  const [bundleInfo, setBundleInfo] = useState<{ size_mb: number; filename: string } | null>(null)
  const [msg, setMsg] = useState<{ type: 'success' | 'error'; text: string } | null>(null)
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null)

  useEffect(() => {
    if (status.has_model) {
      getModelInfo().then((info) => setModelInfo(info)).catch(() => {})
    }
  }, [status.has_model])

  const handlePrepare = async () => {
    setLoading(true)
    setMsg(null)
    try {
      const res = await prepareExportBundle(modelName) as { size_mb: number; filename: string }
      setBundleInfo({ size_mb: res.size_mb, filename: res.filename })
      setReady(true)
      setMsg({ type: 'success', text: 'Export bundle ready.' })
    } catch (e: unknown) {
      setMsg({ type: 'error', text: (e as Error).message })
    } finally {
      setLoading(false)
    }
  }

  if (!status.has_model) {
    return (
      <PageShell title="Export" subtitle="Download your trained model for agents and deployment.">
        <Card>
          <div style={{ textAlign: 'center', padding: '40px 20px', color: theme.textMuted }}>
            <div style={{ fontSize: 15, marginBottom: 6 }}>No model to export</div>
            <div style={{ fontSize: 13 }}>Create and train a model first.</div>
          </div>
        </Card>
      </PageShell>
    )
  }

  const decoderOnly = modelInfo?.decoder_only ?? false

  return (
    <PageShell title="Export" subtitle="Download .bin model + reasoning handler for custom agents.">
      {msg && (
        <div style={{
          background: msg.type === 'success' ? theme.successBg : theme.errorBg,
          border: `1px solid ${msg.type === 'success' ? theme.success : theme.error}`,
          borderRadius: 6, padding: '10px 14px', marginBottom: 16,
          color: msg.type === 'success' ? theme.success : theme.error, fontSize: 13,
        }}>
          {msg.text}
        </div>
      )}

      <Card style={{ marginBottom: 16 }}>
        <label style={labelStyle}>Model name</label>
        <input
          value={modelName}
          onChange={(e) => setModelName(e.target.value.replace(/\s+/g, '_').toLowerCase())}
          placeholder="vnexai_model"
          style={inputStyle}
        />
      </Card>

      {modelInfo && (
        <Card style={{ marginBottom: 16 }}>
          <div style={{ fontWeight: 600, fontSize: 13, color: theme.text, marginBottom: 10 }}>Model info</div>
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
            <Chip label="Type" value={modelInfo.model_type ?? 'Unknown'} />
            <Chip label="Vocab" value={(modelInfo.vocab_size ?? 0).toLocaleString()} />
            {modelInfo.is_trained && modelInfo.final_loss != null && (
              <Chip label="Loss" value={modelInfo.final_loss.toFixed(4)} />
            )}
            <Chip label="CoT ready" value={decoderOnly ? 'Yes' : 'Decoder-only required'} />
          </div>
        </Card>
      )}

      <Card>
        <div style={{ fontWeight: 600, fontSize: 14, color: theme.text, marginBottom: 8 }}>
          Agent export bundle (.zip)
        </div>
        <p style={{ color: theme.textSecondary, fontSize: 13, lineHeight: 1.7, marginBottom: 16 }}>
          Includes <code style={codeStyle}>{modelName}.bin</code>, tokenizer,{' '}
          <code style={codeStyle}>reasoning_handler.py</code>, runtime files, and a README.
          The handler implements the same two-pass thinking loop used in Inference mode — force-open{' '}
          <code style={codeStyle}>&lt;redacted_thinking&gt;</code>, generate steps, self-check, then answer.
        </p>

        <div style={{ background: theme.editor, border: `1px solid ${theme.border}`, borderRadius: 6, padding: '12px 14px', marginBottom: 16 }}>
          <div style={{ fontSize: 11, color: theme.textMuted, fontWeight: 600, marginBottom: 8 }}>BUNDLE CONTENTS</div>
          {[
            `${modelName}.bin`,
            `${modelName}_tokenizer.bin`,
            'reasoning_handler.py',
            'transformer_model.py',
            'chatbot_tokenizer.py',
            'README.md',
          ].map((f) => (
            <div key={f} style={{ fontSize: 12, color: theme.textSecondary, fontFamily: theme.mono, marginBottom: 4 }}>{f}</div>
          ))}
        </div>

        <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
          <Btn variant="primary" onClick={handlePrepare} disabled={loading || !status.is_trained}>
            {loading ? 'Preparing…' : 'Prepare Bundle'}
          </Btn>
          {ready && bundleInfo && (
            <Btn onClick={() => downloadUrl('/export/download-bundle', modelName)}>
              Download {bundleInfo.filename} ({bundleInfo.size_mb} MB)
            </Btn>
          )}
        </div>

        {!status.is_trained && (
          <div style={{ marginTop: 12, fontSize: 12, color: theme.warning }}>
            Train the model before exporting.
          </div>
        )}
        {!decoderOnly && (
          <div style={{ marginTop: 12, fontSize: 12, color: theme.warning }}>
            CoT reasoning handler requires a decoder-only transformer. Architecture export still works.
          </div>
        )}
      </Card>
    </PageShell>
  )
}

function Chip({ label, value }: { label: string; value: string }) {
  return (
    <div style={{ background: theme.editor, borderRadius: 4, padding: '5px 10px', border: `1px solid ${theme.border}` }}>
      <span style={{ fontSize: 11, color: theme.textMuted }}>{label}: </span>
      <span style={{ fontSize: 12, color: theme.text }}>{value}</span>
    </div>
  )
}

const labelStyle: React.CSSProperties = { fontSize: 12, color: theme.textSecondary, display: 'block', marginBottom: 6 }
const inputStyle: React.CSSProperties = {
  width: '100%', background: theme.input, border: `1px solid ${theme.inputBorder}`,
  borderRadius: 4, padding: '7px 10px', color: theme.text, fontSize: 13, outline: 'none', fontFamily: theme.font,
}
const codeStyle: React.CSSProperties = {
  color: theme.info, background: theme.infoBg, padding: '1px 5px', borderRadius: 3, fontFamily: theme.mono, fontSize: 12,
}
