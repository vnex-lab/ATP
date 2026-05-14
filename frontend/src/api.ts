const BASE = '/api'

async function req<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(BASE + path, options)
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(body.detail ?? res.statusText)
  }
  return res.json()
}

function json(method: string, body: unknown): RequestInit {
  return {
    method,
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  }
}

// ── Status ──────────────────────────────────────────────────────────────────
export const getStatus = () => req('/status')

// ── Data ────────────────────────────────────────────────────────────────────
export async function uploadFile(file: File) {
  const fd = new FormData()
  fd.append('file', file)
  return req('/data/upload-file', { method: 'POST', body: fd })
}

export const uploadText = (pairs: { user: string; bot: string }[]) =>
  req('/data/upload-text', json('POST', { pairs }))

export const loadBuiltinAssistant = (target_mb: number) =>
  req('/data/load-builtin', json('POST', { type: 'assistant', target_mb }))

export const loadBuiltinSFT = (rows: number) =>
  req('/data/load-builtin', json('POST', { type: 'sft', rows }))

export const getDataInfo = () => req('/data/info')

// ── Model ────────────────────────────────────────────────────────────────────
export const estimateVocab = (max_vocab_size: number) =>
  req('/model/estimate-vocab', json('POST', { max_vocab_size }))

export const buildVocab = (opts: {
  max_vocab_size: number
  pad_vocab: boolean
  pad_target: number
}) => req('/model/build-vocab', json('POST', opts))

export const createModel = (config: Record<string, unknown>) =>
  req('/model/create', json('POST', config))

// ── Training ─────────────────────────────────────────────────────────────────
export const startTraining = (config: Record<string, unknown>) =>
  req('/training/start', json('POST', config))

export const getTrainingStatus = () => req('/training/status')

export function subscribeToTraining(
  onEvent: (data: unknown) => void,
  onDone?: () => void
): () => void {
  const es = new EventSource('/api/training/stream')
  es.onmessage = (e) => {
    try {
      const data = JSON.parse(e.data)
      onEvent(data)
      if (data.status === 'done' || data.status === 'error') {
        es.close()
        onDone?.()
      }
    } catch {}
  }
  es.onerror = () => {
    es.close()
    onDone?.()
  }
  return () => es.close()
}

// ── Chat ─────────────────────────────────────────────────────────────────────
export const sendChat = (message: string, temperature: number) =>
  req('/chat/send', json('POST', { message, temperature }))

export const clearChat = () => req('/chat/clear', { method: 'POST' })

// ── Export ───────────────────────────────────────────────────────────────────
export const prepareModel = () => req('/export/prepare-model', { method: 'POST' })
export const prepareTokenizer = () => req('/export/prepare-tokenizer', { method: 'POST' })
export const prepareGGUF = (model_name: string) =>
  req('/export/prepare-gguf', json('POST', { model_name }))

export const getModelInfo = () => req<import('./types').ModelInfo>('/export/model-info')

export const generateModelfile = (opts: {
  model_name: string
  ollama_username: string
  system_prompt: string
  num_ctx: number
  temperature: number
  extra_stop: string
}) => req('/export/modelfile', json('POST', opts))

export function downloadUrl(path: string, name: string) {
  const a = document.createElement('a')
  a.href = `/api${path}?name=${encodeURIComponent(name)}`
  a.download = ''
  a.click()
}
