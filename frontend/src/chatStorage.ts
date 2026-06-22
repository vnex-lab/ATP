import type { ChatMessage } from './types'

export interface Conversation {
  id: string
  title: string
  messages: ChatMessage[]
  createdAt: number
  updatedAt: number
}

const STORAGE_KEY = 'vnexai_conversations'
const ACTIVE_KEY = 'vnexai_active_conversation'

function readAll(): Conversation[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (!raw) return []
    const parsed = JSON.parse(raw) as Conversation[]
    return Array.isArray(parsed) ? parsed : []
  } catch {
    return []
  }
}

function writeAll(conversations: Conversation[]) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(conversations))
}

export function listConversations(): Conversation[] {
  return readAll().sort((a, b) => b.updatedAt - a.updatedAt)
}

export function getActiveConversationId(): string | null {
  return localStorage.getItem(ACTIVE_KEY)
}

export function setActiveConversationId(id: string) {
  localStorage.setItem(ACTIVE_KEY, id)
}

export function createConversation(): Conversation {
  const conv: Conversation = {
    id: crypto.randomUUID(),
    title: 'New conversation',
    messages: [],
    createdAt: Date.now(),
    updatedAt: Date.now(),
  }
  const all = readAll()
  all.unshift(conv)
  writeAll(all)
  setActiveConversationId(conv.id)
  return conv
}

export function getConversation(id: string): Conversation | undefined {
  return readAll().find((c) => c.id === id)
}

export function saveConversation(conv: Conversation) {
  const all = readAll()
  const idx = all.findIndex((c) => c.id === conv.id)
  const next = { ...conv, updatedAt: Date.now() }
  if (idx >= 0) all[idx] = next
  else all.unshift(next)
  writeAll(all)
  return next
}

export function deleteConversation(id: string) {
  const all = readAll().filter((c) => c.id !== id)
  writeAll(all)
  if (getActiveConversationId() === id) {
    if (all.length > 0) setActiveConversationId(all[0].id)
    else localStorage.removeItem(ACTIVE_KEY)
  }
}

export function titleFromMessage(text: string): string {
  const trimmed = text.trim().replace(/\s+/g, ' ')
  if (!trimmed) return 'New conversation'
  return trimmed.length > 42 ? `${trimmed.slice(0, 42)}...` : trimmed
}
