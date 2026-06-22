import { useState, useEffect, useCallback } from 'react'
import Sidebar from './components/Sidebar'
import DataPage from './pages/DataPage'
import ModelsPage from './pages/ModelsPage'
import ModelPage from './pages/ModelPage'
import PretrainPage from './pages/PretrainPage'
import TrainPage from './pages/TrainPage'
import ChatPage from './pages/ChatPage'
import ExportPage from './pages/ExportPage'
import { getStatus } from './api'
import type { Page, AppStatus } from './types'
import { theme } from './theme'

const DEFAULT_STATUS: AppStatus = {
  has_training_data: false,
  training_data_count: 0,
  has_tokenizer: false,
  tokenizer_vocab_size: 0,
  has_model: false,
  is_trained: false,
  model_config: null,
  model_type: null,
  training_data_profile: null,
  chat_history: [],
  training_mode: 'scratch',
  active_model_name: null,
  active_model_slug: null,
  saved_models_count: 0,
  cot_reasoning: {
    enabled: false,
    loaded: false,
    count: 0,
    decoder_only: null,
  },
  training: {
    is_training: false,
    progress: 0,
    current_epoch: 0,
    total_epochs: 0,
    current_batch: 0,
    total_batches: 0,
    current_loss: 0,
    avg_loss: 0,
    losses: [],
    val_losses: [],
    status: 'idle',
    error: null,
    gpu_available: false,
  },
}

export default function App() {
  const [page, setPage] = useState<Page>('data')
  const [status, setStatus] = useState<AppStatus>(DEFAULT_STATUS)

  const refreshStatus = useCallback(async () => {
    try {
      const s = await getStatus() as AppStatus
      setStatus(s)
    } catch {}
  }, [])

  useEffect(() => {
    refreshStatus()
    const id = setInterval(refreshStatus, 3000)
    return () => clearInterval(id)
  }, [refreshStatus])

  const isInference = page === 'chat'

  return (
    <div
      className="flex h-screen w-screen overflow-hidden"
      style={{ background: theme.bg, color: theme.text, fontFamily: theme.font }}
    >
      {!isInference && <Sidebar page={page} setPage={setPage} status={status} />}
      <main
        className="flex-1 min-h-0"
        style={{
          background: theme.bg,
          overflow: isInference ? 'hidden' : 'auto',
        }}
      >
        <div className={`fade-in ${isInference ? 'h-full' : ''}`}>
          {page === 'data'     && <DataPage    status={status} onRefresh={refreshStatus} />}
          {page === 'models'   && <ModelsPage  status={status} onRefresh={refreshStatus} />}
          {page === 'model'    && <ModelPage   status={status} onRefresh={refreshStatus} />}
          {page === 'pretrain' && <PretrainPage status={status} onRefresh={refreshStatus} />}
          {page === 'train'    && <TrainPage   status={status} onRefresh={refreshStatus} />}
          {page === 'chat'     && (
            <ChatPage
              status={status}
              onRefresh={refreshStatus}
              onLeave={() => setPage('train')}
            />
          )}
          {page === 'export'   && <ExportPage  status={status} onRefresh={refreshStatus} />}
        </div>
      </main>
      {isInference && (
        <Sidebar page={page} setPage={setPage} status={status} compact />
      )}
    </div>
  )
}
