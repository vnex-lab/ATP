import { useState, useEffect } from 'react'
import type { AppStatus, SavedModel } from '../types'
import {
  listSavedModels,
  saveModel,
  loadSavedModel,
  deleteSavedModel,
  importVnexModel,
  registerExternalModel,
} from '../api'
import PageShell from '../components/PageShell'
import Card from '../components/Card'
import Btn from '../components/Btn'
import FileUpload from '../components/FileUpload'
import { theme } from '../theme'
import { inputStyle, hintStyle, alertColors } from '../styles'

export default function ModelsPage({ status, onRefresh }: { status: AppStatus; onRefresh: () => void }) {
  const [models, setModels] = useState<SavedModel[]>([])
  const [loading, setLoading] = useState(false)
  const [msg, setMsg] = useState<{ type: 'success' | 'error' | 'info'; text: string } | null>(null)
  const [saveName, setSaveName] = useState(status.active_model_name ?? '')
  const [importName, setImportName] = useState('')
  const [importModelFile, setImportModelFile] = useState<File | null>(null)
  const [importTokFile, setImportTokFile] = useState<File | null>(null)
  const [externalName, setExternalName] = useState('')
  const [externalFormat, setExternalFormat] = useState('llama')
  const [externalRef, setExternalRef] = useState('')

  const refresh = async () => {
    try {
      const res = await listSavedModels() as { models: SavedModel[] }
      setModels(res.models)
    } catch {}
  }

  useEffect(() => { refresh() }, [])
  useEffect(() => {
    if (status.active_model_name) setSaveName(status.active_model_name)
  }, [status.active_model_name])

  const handleSave = async () => {
    if (!saveName.trim()) return setMsg({ type: 'error', text: 'Enter a model name.' })
    setLoading(true)
    setMsg(null)
    try {
      const res = await saveModel(saveName.trim()) as { message: string }
      setMsg({ type: 'success', text: res.message })
      await refresh()
      onRefresh()
    } catch (e: unknown) {
      setMsg({ type: 'error', text: (e as Error).message })
    } finally {
      setLoading(false)
    }
  }

  const handleLoad = async (slug: string) => {
    setLoading(true)
    setMsg(null)
    try {
      const res = await loadSavedModel(slug) as { message: string }
      setMsg({ type: 'success', text: res.message })
      onRefresh()
    } catch (e: unknown) {
      setMsg({ type: 'error', text: (e as Error).message })
    } finally {
      setLoading(false)
    }
  }

  const handleDelete = async (slug: string) => {
    setLoading(true)
    try {
      await deleteSavedModel(slug)
      setMsg({ type: 'info', text: 'Model deleted.' })
      await refresh()
      onRefresh()
    } catch (e: unknown) {
      setMsg({ type: 'error', text: (e as Error).message })
    } finally {
      setLoading(false)
    }
  }

  const handleImport = async () => {
    if (!importName.trim() || !importModelFile) {
      return setMsg({ type: 'error', text: 'Model name and weights file are required.' })
    }
    setLoading(true)
    setMsg(null)
    try {
      const res = await importVnexModel(importName.trim(), importModelFile, importTokFile) as { message: string }
      setMsg({ type: 'success', text: res.message })
      setImportName('')
      setImportModelFile(null)
      setImportTokFile(null)
      await refresh()
      onRefresh()
    } catch (e: unknown) {
      setMsg({ type: 'error', text: (e as Error).message })
    } finally {
      setLoading(false)
    }
  }

  const handleRegisterExternal = async () => {
    if (!externalName.trim()) return setMsg({ type: 'error', text: 'Enter a name for the external model.' })
    setLoading(true)
    try {
      const res = await registerExternalModel({
        name: externalName.trim(),
        format: externalFormat,
        reference: externalRef.trim(),
        note: 'Registered for library tracking. External models use their own tokenizer (e.g. Llama, Mistral).',
      }) as { message: string }
      setMsg({ type: 'success', text: res.message })
      setExternalName('')
      setExternalRef('')
      await refresh()
    } catch (e: unknown) {
      setMsg({ type: 'error', text: (e as Error).message })
    } finally {
      setLoading(false)
    }
  }

  return (
    <PageShell
      title="Model Library"
      subtitle="Save, load, import, and fine-tune models. Training auto-saves after each run."
    >
      {msg && <Alert type={msg.type} text={msg.text} />}

      <Card style={{ marginBottom: 16 }}>
        <SectionTitle>Save current model</SectionTitle>
        <p style={hintStyle}>
          Every completed training run is auto-saved. You can also save manually with a custom name.
        </p>
        <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', alignItems: 'center' }}>
          <input
            value={saveName}
            onChange={(e) => setSaveName(e.target.value)}
            placeholder="my_assistant_v1"
            style={{ ...inputStyle, flex: 1, minWidth: 200 }}
            disabled={!status.has_model || loading}
          />
          <Btn variant="primary" onClick={handleSave} disabled={!status.has_model || loading}>
            Save Model
          </Btn>
        </div>
        {status.active_model_name && (
          <div style={{ marginTop: 8, fontSize: 12, color: theme.textMuted }}>
            Active: {status.active_model_name}
            {status.training_mode === 'finetune' ? ' · fine-tune mode' : ' · scratch mode'}
          </div>
        )}
      </Card>

      <Card style={{ marginBottom: 16 }}>
        <SectionTitle>Import model weights</SectionTitle>
        <p style={hintStyle}>
          Import VnexAI .bin weights for fine-tuning. A tokenizer is optional — models like Llama, Mistral, and
          other external checkpoints ship with their own tokenizer and do not need a VnexAI tokenizer file here.
        </p>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginBottom: 12 }}>
          <Field label="Model name">
            <input value={importName} onChange={(e) => setImportName(e.target.value)} placeholder="llama_style_v1" style={inputStyle} />
          </Field>
          <FileUpload
            label="Model weights"
            accept=".bin,.gguf,.safetensors"
            onFileChange={setImportModelFile}
            disabled={loading}
          />
          <FileUpload
            label="VnexAI tokenizer"
            accept=".bin"
            optional
            onFileChange={setImportTokFile}
            disabled={loading}
          />
        </div>
        <Btn variant="primary" onClick={handleImport} disabled={loading || !importModelFile}>
          Import Model
        </Btn>
      </Card>

      <Card style={{ marginBottom: 16 }}>
        <SectionTitle>Register external model</SectionTitle>
        <p style={hintStyle}>
          Track models like Llama 3.2, Mistral, or other GGUF/HF checkpoints in your library without uploading files.
        </p>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 12, marginBottom: 12 }}>
          <Field label="Display name">
            <input value={externalName} onChange={(e) => setExternalName(e.target.value)} placeholder="llama3.2" style={inputStyle} />
          </Field>
          <Field label="Format">
            <select value={externalFormat} onChange={(e) => setExternalFormat(e.target.value)} style={inputStyle}>
              <option value="llama">Llama / GGUF</option>
              <option value="huggingface">HuggingFace</option>
              <option value="ollama">Ollama</option>
              <option value="other">Other</option>
            </select>
          </Field>
          <Field label="Reference (path or repo)">
            <input value={externalRef} onChange={(e) => setExternalRef(e.target.value)} placeholder="meta-llama/Llama-3.2-3B" style={inputStyle} />
          </Field>
        </div>
        <Btn onClick={handleRegisterExternal} disabled={loading}>Register External Model</Btn>
      </Card>

      <Card>
        <SectionTitle>Saved models ({models.length})</SectionTitle>
        {models.length === 0 ? (
          <div style={{ color: theme.textMuted, fontSize: 13, padding: '12px 0' }}>
            No saved models yet. Train a model or import weights.
          </div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {models.map((m) => (
              <div
                key={m.slug}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 12,
                  padding: '12px 14px',
                  background: theme.editor,
                  border: `1px solid ${status.active_model_slug === m.slug ? theme.accent : theme.border}`,
                  borderRadius: 6,
                }}
              >
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ color: theme.text, fontWeight: 600, fontSize: 13 }}>{m.name}</div>
                  <div style={{ color: theme.textMuted, fontSize: 11, marginTop: 2 }}>
                    {m.source ?? 'saved'}
                    {m.model_type ? ` · ${m.model_type}` : ''}
                    {m.tokenizer_missing ? ' · no tokenizer' : ''}
                    {m.finetune_ready === false ? ' · reference only' : ' · fine-tune ready'}
                    {m.vocab_size ? ` · ${m.vocab_size.toLocaleString()} vocab` : ''}
                  </div>
                </div>
                {m.finetune_ready !== false && m.exists !== false && (
                  <Btn onClick={() => handleLoad(m.slug)} disabled={loading}>Load</Btn>
                )}
                <Btn variant="danger" onClick={() => handleDelete(m.slug)} disabled={loading}>Delete</Btn>
              </div>
            ))}
          </div>
        )}
      </Card>
    </PageShell>
  )
}

function SectionTitle({ children }: { children: React.ReactNode }) {
  return <div style={{ fontWeight: 600, fontSize: 14, color: theme.text, marginBottom: 8 }}>{children}</div>
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <label style={{ fontSize: 12, color: theme.textSecondary, display: 'block', marginBottom: 6 }}>{label}</label>
      {children}
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
