export type Page = 'data' | 'models' | 'model' | 'pretrain' | 'train' | 'chat' | 'export'

export interface Conversation {
  id: string
  title: string
  messages: ChatMessage[]
  createdAt: number
  updatedAt: number
}

export interface TrainingPair {
  user: string
  bot: string
}

export interface TrainingState {
  is_training: boolean
  progress: number
  current_epoch: number
  total_epochs: number
  current_batch: number
  total_batches: number
  current_loss: number
  avg_loss: number
  losses: number[]
  val_losses: number[]
  status: 'idle' | 'training' | 'done' | 'error'
  error: string | null
  gpu_available: boolean
}

export interface PretrainStatus {
  loaded: boolean
  count: number
  persisted: boolean
}

export interface CotReasoningStatus {
  enabled: boolean
  loaded: boolean
  count: number
  decoder_only: boolean | null
}

export interface SavedModel {
  slug: string
  name: string
  source?: string
  model_type?: string
  vocab_size?: number
  finetune_ready?: boolean
  exists?: boolean
  updated_at?: string
  note?: string
  format?: string
  reference?: string
  tokenizer_missing?: boolean
}

export interface AppStatus {
  has_training_data: boolean
  training_data_count: number
  has_tokenizer: boolean
  tokenizer_vocab_size: number
  has_model: boolean
  is_trained: boolean
  model_config: Record<string, unknown> | null
  model_type: string | null
  training_data_profile: string | null
  chat_history: ChatMessage[]
  training: TrainingState
  cot_reasoning?: CotReasoningStatus
  training_mode?: 'scratch' | 'finetune'
  active_model_name?: string | null
  active_model_slug?: string | null
  saved_models_count?: number
}

export interface ChatMessage {
  role: 'user' | 'bot'
  content: string
  reasoning?: string
}

export interface ModelInfo {
  has_model: boolean
  is_rnn?: boolean
  is_trained?: boolean
  model_type?: string
  vocab_size?: number
  embedding_dim?: number
  hidden_dim?: number
  max_length?: number
  num_heads?: number
  num_layers?: number
  ff_dim?: number
  max_seq_len?: number
  learning_rate?: number
  decoder_only?: boolean
  final_loss?: number
  loss_history?: number[]
}
