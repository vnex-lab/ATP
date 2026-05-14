import { useState, useEffect, useCallback } from 'react'
import Sidebar from './components/Sidebar'
import DataPage from './pages/DataPage'
import ModelPage from './pages/ModelPage'
import TrainPage from './pages/TrainPage'
import ChatPage from './pages/ChatPage'
import ExportPage from './pages/ExportPage'
import { getStatus } from './api'
import type { Page, AppStatus } from './types'

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

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-[#080808]">
      <Sidebar page={page} setPage={setPage} status={status} />
      <main className="flex-1 overflow-y-auto">
        <div className="fade-in h-full">
          {page === 'data'   && <DataPage  status={status} onRefresh={refreshStatus} />}
          {page === 'model'  && <ModelPage status={status} onRefresh={refreshStatus} />}
          {page === 'train'  && <TrainPage status={status} onRefresh={refreshStatus} />}
          {page === 'chat'   && <ChatPage  status={status} onRefresh={refreshStatus} />}
          {page === 'export' && <ExportPage status={status} onRefresh={refreshStatus} />}
        </div>
      </main>
    </div>
  )
}
