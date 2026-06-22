import { useState, useRef, useEffect, useCallback } from 'react'
import { Plus, Send, Settings, Trash2, Copy, MessageSquare, ChevronLeft } from 'lucide-react'
import type { AppStatus, ChatMessage, SavedModel } from '../types'
import { sendChat, clearChat, restoreChat, setCotReasoning, listSavedModels, loadSavedModel } from '../api'
import {
  listConversations,
  createConversation,
  getActiveConversationId,
  setActiveConversationId,
  saveConversation,
  deleteConversation,
  getConversation,
  titleFromMessage,
} from '../chatStorage'
import Btn from '../components/Btn'
import { theme } from '../theme'

interface Props {
  status: AppStatus
  onRefresh: () => void
  onLeave?: () => void
}

export default function ChatPage({ status, onRefresh, onLeave }: Props) {
  const [input, setInput] = useState('')
  const [temperature, setTemperature] = useState(0.8)
  const [loading, setLoading] = useState(false)
  const [msgs, setMsgs] = useState<ChatMessage[]>([])
  const [err, setErr] = useState<string | null>(null)
  const [showSettings, setShowSettings] = useState(false)
  const [conversations, setConversations] = useState(() => listConversations())
  const [activeId, setActiveId] = useState<string | null>(() => getActiveConversationId())
  const [savedModels, setSavedModels] = useState<SavedModel[]>([])
  const [modelLoading, setModelLoading] = useState(false)

  const bottomRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)

  const [reasoningEnabled, setReasoningEnabled] = useState(status.cot_reasoning?.enabled ?? false)
  const cotLoaded = status.cot_reasoning?.loaded ?? false
  const decoderOnly = status.cot_reasoning?.decoder_only ?? false
  const canReason = decoderOnly && cotLoaded

  const modelLabel = status.active_model_name ?? status.model_type ?? 'Active model'
  const ready = status.is_trained

  useEffect(() => {
    setReasoningEnabled(status.cot_reasoning?.enabled ?? false)
  }, [status.cot_reasoning?.enabled])

  useEffect(() => {
    listSavedModels()
      .then((r) => setSavedModels(r.models ?? []))
      .catch(() => {})
  }, [status.saved_models_count, status.active_model_slug])

  useEffect(() => {
    if (!activeId) {
      const conv = createConversation()
      setActiveId(conv.id)
      setConversations(listConversations())
      setMsgs([])
      return
    }
    const conv = getConversation(activeId)
    if (conv) {
      setMsgs(conv.messages)
      if (ready) restoreChat(conv.messages).catch(() => {})
    }
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [msgs, loading])

  const persistCurrent = useCallback((messages: ChatMessage[], title?: string) => {
    if (!activeId) return
    const existing = getConversation(activeId)
    if (!existing) return
    const next = saveConversation({
      ...existing,
      messages,
      title: title ?? existing.title,
    })
    setConversations(listConversations())
    return next
  }, [activeId])

  const selectConversation = async (id: string) => {
    if (activeId) persistCurrent(msgs)
    setActiveId(id)
    setActiveConversationId(id)
    const conv = getConversation(id)
    const messages = conv?.messages ?? []
    setMsgs(messages)
    setErr(null)
    if (ready) {
      try {
        await restoreChat(messages)
        onRefresh()
      } catch {}
    }
  }

  const handleNewChat = async () => {
    if (activeId) persistCurrent(msgs)
    const conv = createConversation()
    setConversations(listConversations())
    setActiveId(conv.id)
    setMsgs([])
    setErr(null)
    setInput('')
    if (ready) {
      try {
        await clearChat()
        onRefresh()
      } catch {}
    }
    setTimeout(() => inputRef.current?.focus(), 50)
  }

  const handleDeleteConversation = async (id: string, e: React.MouseEvent) => {
    e.stopPropagation()
    deleteConversation(id)
    const remaining = listConversations()
    setConversations(remaining)
    if (activeId === id) {
      if (remaining.length > 0) {
        await selectConversation(remaining[0].id)
      } else {
        await handleNewChat()
      }
    }
  }

  const toggleReasoning = async () => {
    if (!canReason) return
    const next = !reasoningEnabled
    try {
      await setCotReasoning(next)
      setReasoningEnabled(next)
      onRefresh()
    } catch (e: unknown) {
      setErr((e as Error).message)
    }
  }

  const send = async () => {
    const text = input.trim()
    if (!text || loading || !ready) return
    setErr(null)
    setInput('')
    const optimistic: ChatMessage[] = [...msgs, { role: 'user', content: text }]
    setMsgs(optimistic)
    if (msgs.length === 0) {
      persistCurrent(optimistic, titleFromMessage(text))
    }
    setLoading(true)
    try {
      const res = await sendChat(text, temperature)
      setMsgs(res.history)
      persistCurrent(res.history)
      onRefresh()
    } catch (e: unknown) {
      setErr((e as Error).message)
      persistCurrent(optimistic)
    } finally {
      setLoading(false)
      setTimeout(() => inputRef.current?.focus(), 50)
    }
  }

  const handleLoadModel = async (slug: string) => {
    if (modelLoading || slug === status.active_model_slug) return
    setModelLoading(true)
    setErr(null)
    try {
      await loadSavedModel(slug)
      onRefresh()
    } catch (e: unknown) {
      setErr((e as Error).message)
    } finally {
      setModelLoading(false)
    }
  }

  const copyText = (text: string) => {
    navigator.clipboard.writeText(text).catch(() => {})
  }

  const tempLabel = temperature < 0.5 ? 'Focused' : temperature > 1.5 ? 'Creative' : 'Balanced'

  if (!ready) {
    return (
      <div style={shellStyle}>
        <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 32 }}>
          <div style={{ textAlign: 'center', maxWidth: 420 }}>
            <MessageSquare size={40} color={theme.textDim} style={{ margin: '0 auto 16px' }} />
            <div style={{ fontSize: 16, fontWeight: 600, color: theme.text, marginBottom: 8 }}>No trained model loaded</div>
            <div style={{ fontSize: 13, color: theme.textMuted, lineHeight: 1.6, marginBottom: 20 }}>
              Train a model in the Studio workflow, or load one from the Model Library to start inference.
            </div>
            {onLeave && <Btn variant="primary" onClick={onLeave}>Go to Studio</Btn>}
          </div>
        </div>
      </div>
    )
  }

  return (
    <div style={shellStyle}>
      {/* Conversation sidebar */}
      <aside style={convSidebarStyle}>
        <div style={{ padding: '12px 10px', borderBottom: `1px solid ${theme.border}` }}>
          <button onClick={handleNewChat} style={newChatBtnStyle}>
            <Plus size={15} />
            New conversation
          </button>
        </div>
        <div style={{ flex: 1, overflowY: 'auto', padding: '6px 8px' }}>
          {conversations.map((c) => {
            const active = c.id === activeId
            return (
              <button
                key={c.id}
                onClick={() => selectConversation(c.id)}
                style={{
                  ...convItemStyle,
                  background: active ? theme.active : 'transparent',
                  color: active ? '#fff' : theme.textSecondary,
                }}
              >
                <MessageSquare size={14} style={{ flexShrink: 0, opacity: 0.7 }} />
                <span style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', textAlign: 'left' }}>
                  {c.title}
                </span>
                <span
                  role="button"
                  tabIndex={0}
                  onClick={(e) => handleDeleteConversation(c.id, e)}
                  onKeyDown={(e) => { if (e.key === 'Enter') handleDeleteConversation(c.id, e as unknown as React.MouseEvent) }}
                  style={{ opacity: 0.5, padding: 2, display: 'flex' }}
                >
                  <Trash2 size={13} />
                </span>
              </button>
            )
          })}
        </div>
        {onLeave && (
          <div style={{ padding: '10px 10px 12px', borderTop: `1px solid ${theme.border}` }}>
            <button onClick={onLeave} style={studioLinkStyle}>
              <ChevronLeft size={14} />
              Back to Studio
            </button>
          </div>
        )}
      </aside>

      {/* Main chat area */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minWidth: 0 }}>
        {/* Top bar */}
        <header style={topBarStyle}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12, minWidth: 0 }}>
            <span style={{ fontWeight: 600, fontSize: 14, color: theme.text }}>Inference</span>
            <span style={{ color: theme.textDim }}>|</span>
            <select
              value={status.active_model_slug ?? ''}
              onChange={(e) => handleLoadModel(e.target.value)}
              disabled={modelLoading || savedModels.length === 0}
              style={modelSelectStyle}
            >
              <option value={status.active_model_slug ?? ''}>{modelLabel}</option>
              {savedModels
                .filter((m) => m.slug !== status.active_model_slug)
                .map((m) => (
                  <option key={m.slug} value={m.slug}>{m.name}</option>
                ))}
            </select>
            {status.training_mode === 'finetune' && (
              <span style={tagStyle}>Fine-tuned</span>
            )}
          </div>
          <button
            onClick={() => setShowSettings((s) => !s)}
            style={{
              ...iconBtnStyle,
              background: showSettings ? theme.active : 'transparent',
              color: showSettings ? '#fff' : theme.textSecondary,
            }}
            title="Generation settings"
          >
            <Settings size={16} />
          </button>
        </header>

        {showSettings && (
          <div style={settingsBarStyle}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 12, flex: 1, minWidth: 200 }}>
              <label style={settingsLabelStyle}>Temperature</label>
              <input
                type="range"
                min={0.1}
                max={2.0}
                step={0.1}
                value={temperature}
                onChange={(e) => setTemperature(Number(e.target.value))}
                style={{ flex: 1, accentColor: theme.accent }}
              />
              <span style={{ fontSize: 12, color: theme.text, minWidth: 28 }}>{temperature.toFixed(1)}</span>
              <span style={tagStyle}>{tempLabel}</span>
            </div>
            <button
              onClick={toggleReasoning}
              disabled={!canReason}
              title={
                !decoderOnly
                  ? 'Requires a decoder-only transformer'
                  : !cotLoaded
                    ? 'Load CoT data on the Pre-train page first'
                    : reasoningEnabled
                      ? 'Disable chain-of-thought reasoning'
                      : 'Enable chain-of-thought reasoning'
              }
              style={{
                ...toggleStyle,
                borderColor: reasoningEnabled && canReason ? theme.accent : theme.border,
                background: reasoningEnabled && canReason ? theme.accentMuted : theme.panel,
                color: !canReason ? theme.textDim : reasoningEnabled ? theme.info : theme.textSecondary,
                cursor: canReason ? 'pointer' : 'default',
              }}
            >
              CoT reasoning: {reasoningEnabled && canReason ? 'On' : 'Off'}
            </button>
          </div>
        )}

        {/* Messages */}
        <div style={messagesStyle}>
          {msgs.length === 0 && (
            <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', color: theme.textDim }}>
              <div style={{ fontSize: 15, fontWeight: 500, color: theme.textMuted, marginBottom: 6 }}>Start a conversation</div>
              <div style={{ fontSize: 13 }}>Send a message to run inference on your model.</div>
            </div>
          )}
          {msgs.map((msg, i) => (
            <MessageBubble key={i} msg={msg} onCopy={copyText} />
          ))}
          {loading && (
            <div style={{ display: 'flex', gap: 10, alignItems: 'flex-start' }}>
              <RoleBadge label="Model" />
              <div style={{ ...bubbleStyle, background: theme.panel, borderColor: theme.border }}>
                <div style={{ display: 'flex', gap: 5, padding: '4px 0' }}>
                  {[0, 0.2, 0.4].map((delay) => (
                    <span
                      key={delay}
                      className="pulse-dot"
                      style={{ width: 6, height: 6, borderRadius: '50%', background: theme.textDim, display: 'inline-block', animationDelay: `${delay}s` }}
                    />
                  ))}
                </div>
              </div>
            </div>
          )}
          {err && (
            <div style={{ background: theme.errorBg, border: `1px solid ${theme.error}`, borderRadius: 6, padding: '10px 14px', color: theme.error, fontSize: 13 }}>
              {err}
            </div>
          )}
          <div ref={bottomRef} />
        </div>

        {/* Input */}
        <div style={inputBarStyle}>
          <textarea
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault()
                send()
              }
            }}
            placeholder="Message the model... (Enter to send, Shift+Enter for new line)"
            disabled={loading}
            rows={1}
            style={textareaStyle}
          />
          <button
            onClick={send}
            disabled={loading || !input.trim()}
            style={{
              ...sendBtnStyle,
              background: loading || !input.trim() ? theme.panel : theme.accent,
              color: loading || !input.trim() ? theme.textDim : '#fff',
              cursor: loading || !input.trim() ? 'default' : 'pointer',
            }}
          >
            <Send size={16} />
          </button>
        </div>
      </div>
    </div>
  )
}

function MessageBubble({ msg, onCopy }: { msg: ChatMessage; onCopy: (t: string) => void }) {
  const isUser = msg.role === 'user'
  return (
    <div style={{ display: 'flex', gap: 10, alignItems: 'flex-start', flexDirection: isUser ? 'row-reverse' : 'row' }}>
      <RoleBadge label={isUser ? 'You' : 'Model'} user={isUser} />
      <div style={{ maxWidth: '72%', position: 'relative' }}>
        <div
          style={{
            ...bubbleStyle,
            background: isUser ? theme.accentMuted : theme.panel,
            borderColor: isUser ? theme.accent : theme.border,
          }}
        >
          {msg.reasoning && (
            <div style={reasoningStyle}>
              <div style={reasoningHeaderStyle}>Reasoning</div>
              {msg.reasoning}
            </div>
          )}
          <div style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-word', fontSize: 14, lineHeight: 1.55, color: theme.text }}>
            {msg.content}
          </div>
        </div>
        <button
          onClick={() => onCopy(msg.content)}
          style={copyBtnStyle}
          title="Copy message"
        >
          <Copy size={12} />
        </button>
      </div>
    </div>
  )
}

function RoleBadge({ label, user }: { label: string; user?: boolean }) {
  return (
    <div
      style={{
        width: 32,
        height: 32,
        borderRadius: 4,
        background: user ? theme.accentMuted : theme.badge,
        border: `1px solid ${user ? theme.accent : theme.border}`,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        fontSize: 10,
        fontWeight: 700,
        color: user ? theme.info : theme.textSecondary,
        flexShrink: 0,
        letterSpacing: '0.02em',
      }}
    >
      {label.slice(0, 1)}
    </div>
  )
}

const shellStyle: React.CSSProperties = {
  display: 'flex',
  height: '100%',
  overflow: 'hidden',
  background: theme.bg,
}

const convSidebarStyle: React.CSSProperties = {
  width: 240,
  minWidth: 240,
  background: theme.sidebar,
  borderRight: `1px solid ${theme.border}`,
  display: 'flex',
  flexDirection: 'column',
  height: '100%',
}

const newChatBtnStyle: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  gap: 8,
  width: '100%',
  padding: '8px 12px',
  borderRadius: 4,
  border: `1px solid ${theme.border}`,
  background: theme.panel,
  color: theme.text,
  fontSize: 13,
  cursor: 'pointer',
  fontFamily: theme.font,
}

const convItemStyle: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  gap: 8,
  width: '100%',
  padding: '8px 10px',
  borderRadius: 4,
  border: 'none',
  fontSize: 12,
  cursor: 'pointer',
  fontFamily: theme.font,
  marginBottom: 2,
}

const studioLinkStyle: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  gap: 6,
  width: '100%',
  padding: '7px 10px',
  borderRadius: 4,
  border: 'none',
  background: 'transparent',
  color: theme.textMuted,
  fontSize: 12,
  cursor: 'pointer',
  fontFamily: theme.font,
}

const topBarStyle: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'space-between',
  padding: '10px 16px',
  borderBottom: `1px solid ${theme.border}`,
  background: theme.sidebar,
  minHeight: 48,
}

const modelSelectStyle: React.CSSProperties = {
  background: theme.input,
  border: `1px solid ${theme.border}`,
  borderRadius: 4,
  padding: '4px 8px',
  color: theme.text,
  fontSize: 12,
  maxWidth: 220,
  fontFamily: theme.font,
}

const tagStyle: React.CSSProperties = {
  fontSize: 11,
  color: theme.textMuted,
  background: theme.badge,
  padding: '2px 8px',
  borderRadius: 3,
  whiteSpace: 'nowrap',
}

const iconBtnStyle: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  width: 32,
  height: 32,
  borderRadius: 4,
  border: 'none',
  cursor: 'pointer',
  fontFamily: theme.font,
}

const settingsBarStyle: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  gap: 16,
  flexWrap: 'wrap',
  padding: '10px 16px',
  borderBottom: `1px solid ${theme.border}`,
  background: theme.panel,
}

const settingsLabelStyle: React.CSSProperties = {
  fontSize: 12,
  color: theme.textMuted,
  whiteSpace: 'nowrap',
}

const toggleStyle: React.CSSProperties = {
  padding: '6px 12px',
  borderRadius: 4,
  border: '1px solid',
  fontSize: 12,
  fontFamily: theme.font,
}

const messagesStyle: React.CSSProperties = {
  flex: 1,
  overflowY: 'auto',
  padding: '20px 24px',
  display: 'flex',
  flexDirection: 'column',
  gap: 16,
}

const bubbleStyle: React.CSSProperties = {
  padding: '12px 14px',
  borderRadius: 6,
  border: '1px solid',
}

const reasoningStyle: React.CSSProperties = {
  marginBottom: 10,
  padding: '8px 10px',
  background: theme.editor,
  border: `1px solid ${theme.accentMuted}`,
  borderRadius: 4,
  fontSize: 12,
  color: theme.info,
  lineHeight: 1.5,
}

const reasoningHeaderStyle: React.CSSProperties = {
  fontSize: 10,
  fontWeight: 700,
  letterSpacing: '0.06em',
  textTransform: 'uppercase',
  marginBottom: 6,
  color: theme.accent,
}

const copyBtnStyle: React.CSSProperties = {
  position: 'absolute',
  top: 6,
  right: 6,
  background: theme.panel,
  border: `1px solid ${theme.border}`,
  borderRadius: 4,
  padding: 4,
  color: theme.textMuted,
  cursor: 'pointer',
  opacity: 0.6,
}

const inputBarStyle: React.CSSProperties = {
  padding: '12px 16px',
  borderTop: `1px solid ${theme.border}`,
  display: 'flex',
  gap: 10,
  alignItems: 'flex-end',
  background: theme.sidebar,
}

const textareaStyle: React.CSSProperties = {
  flex: 1,
  background: theme.input,
  border: `1px solid ${theme.border}`,
  borderRadius: 4,
  padding: '10px 12px',
  color: theme.text,
  fontSize: 14,
  outline: 'none',
  fontFamily: theme.font,
  resize: 'none',
  minHeight: 44,
  maxHeight: 160,
  lineHeight: 1.5,
}

const sendBtnStyle: React.CSSProperties = {
  width: 40,
  height: 40,
  borderRadius: 4,
  border: 'none',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  flexShrink: 0,
}
