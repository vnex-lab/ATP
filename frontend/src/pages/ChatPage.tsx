import { useState, useRef, useEffect } from 'react'
import type { AppStatus, ChatMessage } from '../types'
import { sendChat, clearChat } from '../api'
import PageShell from '../components/PageShell'
import Card from '../components/Card'
import Btn from '../components/Btn'

export default function ChatPage({ status, onRefresh }: { status: AppStatus; onRefresh: () => void }) {
  const [input, setInput] = useState('')
  const [temperature, setTemperature] = useState(0.8)
  const [loading, setLoading] = useState(false)
  const [msgs, setMsgs] = useState<ChatMessage[]>(status.chat_history)
  const [err, setErr] = useState<string | null>(null)
  const bottomRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    setMsgs(status.chat_history)
  }, [status.chat_history])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [msgs, loading])

  const send = async () => {
    const text = input.trim()
    if (!text || loading) return
    setErr(null)
    setInput('')
    setLoading(true)
    try {
      const res = await sendChat(text, temperature) as { response: string; history: ChatMessage[] }
      setMsgs(res.history)
      onRefresh()
    } catch (e: unknown) {
      setErr((e as Error).message)
      setMsgs((prev) => [...prev, { role: 'user', content: text }])
    } finally {
      setLoading(false)
      setTimeout(() => inputRef.current?.focus(), 50)
    }
  }

  const handleClear = async () => {
    await clearChat()
    setMsgs([])
    setErr(null)
    onRefresh()
  }

  const tempLabel = temperature < 0.5 ? 'Focused' : temperature > 1.5 ? 'Creative' : 'Balanced'
  const tempColor = temperature < 0.5 ? '#3b82f6' : temperature > 1.5 ? '#f59e0b' : '#22c55e'

  if (!status.is_trained) {
    return (
      <PageShell title="Chat" subtitle="Test your trained model in a conversation.">
        <Card>
          <div style={{ textAlign: 'center', padding: '40px 20px', color: '#475569' }}>
            <div style={{ fontSize: 40, marginBottom: 12 }}>🤖</div>
            <div style={{ fontSize: 15, color: '#94a3b8', marginBottom: 6 }}>Model not trained yet</div>
            <div style={{ fontSize: 13 }}>Complete the Training step first to start chatting.</div>
          </div>
        </Card>
      </PageShell>
    )
  }

  return (
    <PageShell title="Chat" subtitle="Test your trained model in a conversation.">
      {/* Settings bar */}
      <Card style={{ marginBottom: 12 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 20, flexWrap: 'wrap' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12, flex: 1, minWidth: 200 }}>
            <label style={{ fontSize: 12, color: '#94a3b8', fontWeight: 500, whiteSpace: 'nowrap' }}>
              Temperature
            </label>
            <input
              type="range"
              min={0.1}
              max={2.0}
              step={0.1}
              value={temperature}
              onChange={(e) => setTemperature(Number(e.target.value))}
              style={{ flex: 1, accentColor: tempColor }}
            />
            <span style={{ fontSize: 13, color: tempColor, fontWeight: 600, minWidth: 38, textAlign: 'right' }}>
              {temperature.toFixed(1)}
            </span>
            <span style={{ fontSize: 11, color: tempColor, background: tempColor + '22', padding: '2px 8px', borderRadius: 20, whiteSpace: 'nowrap' }}>
              {tempLabel}
            </span>
          </div>
          {msgs.length > 0 && (
            <Btn onClick={handleClear}>Clear Chat</Btn>
          )}
        </div>
      </Card>

      {/* Chat window */}
      <div style={{
        flex: 1,
        background: '#0f0f0f',
        border: '1px solid #1e1e1e',
        borderRadius: 10,
        display: 'flex',
        flexDirection: 'column',
        height: 'calc(100vh - 280px)',
        minHeight: 300,
        overflow: 'hidden',
      }}>
        {/* Messages */}
        <div style={{ flex: 1, overflowY: 'auto', padding: '16px 20px', display: 'flex', flexDirection: 'column', gap: 12 }}>
          {msgs.length === 0 && (
            <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', color: '#374151' }}>
              <div style={{ fontSize: 32, marginBottom: 8 }}>💬</div>
              <div style={{ fontSize: 14 }}>Send a message to start chatting</div>
            </div>
          )}
          {msgs.map((msg, i) => (
            <div
              key={i}
              style={{
                display: 'flex',
                justifyContent: msg.role === 'user' ? 'flex-end' : 'flex-start',
                gap: 8,
                alignItems: 'flex-end',
              }}
            >
              {msg.role === 'bot' && (
                <div style={{
                  width: 28, height: 28, borderRadius: 8,
                  background: 'linear-gradient(135deg, #3b82f6, #6366f1)',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  fontSize: 13, flexShrink: 0,
                }}>🤖</div>
              )}
              <div style={{
                maxWidth: '70%',
                padding: '10px 14px',
                borderRadius: msg.role === 'user' ? '14px 14px 4px 14px' : '14px 14px 14px 4px',
                background: msg.role === 'user' ? '#1e3a5f' : '#1c1c1c',
                border: `1px solid ${msg.role === 'user' ? '#2563eb44' : '#252525'}`,
                color: '#e2e8f0',
                fontSize: 14,
                lineHeight: 1.5,
                whiteSpace: 'pre-wrap',
                wordBreak: 'break-word',
              }}>
                {msg.content}
              </div>
              {msg.role === 'user' && (
                <div style={{
                  width: 28, height: 28, borderRadius: 8,
                  background: '#1e3a5f', border: '1px solid #2563eb44',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  fontSize: 13, flexShrink: 0,
                }}>👤</div>
              )}
            </div>
          ))}
          {loading && (
            <div style={{ display: 'flex', alignItems: 'flex-end', gap: 8 }}>
              <div style={{ width: 28, height: 28, borderRadius: 8, background: 'linear-gradient(135deg, #3b82f6, #6366f1)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 13 }}>🤖</div>
              <div style={{ background: '#1c1c1c', border: '1px solid #252525', borderRadius: '14px 14px 14px 4px', padding: '10px 16px' }}>
                <div style={{ display: 'flex', gap: 5, alignItems: 'center' }}>
                  {[0, 0.2, 0.4].map((delay) => (
                    <span key={delay} style={{
                      width: 6, height: 6, borderRadius: '50%', background: '#475569',
                      display: 'inline-block', animation: `pulseDot 1.4s ease-in-out ${delay}s infinite`,
                    }} />
                  ))}
                </div>
              </div>
            </div>
          )}
          {err && (
            <div style={{ background: '#450a0a', border: '1px solid #7f1d1d', borderRadius: 8, padding: '8px 12px', color: '#fca5a5', fontSize: 13 }}>
              Error: {err}
            </div>
          )}
          <div ref={bottomRef} />
        </div>

        {/* Input */}
        <div style={{ padding: '12px 16px', borderTop: '1px solid #1e1e1e', display: 'flex', gap: 10 }}>
          <input
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send() } }}
            placeholder="Type a message… (Enter to send)"
            disabled={loading}
            style={{
              flex: 1,
              background: '#111',
              border: '1px solid #252525',
              borderRadius: 8,
              padding: '9px 14px',
              color: '#e2e8f0',
              fontSize: 14,
              outline: 'none',
              fontFamily: 'inherit',
              opacity: loading ? 0.7 : 1,
            }}
          />
          <button
            onClick={send}
            disabled={loading || !input.trim()}
            style={{
              width: 40,
              height: 40,
              borderRadius: 8,
              border: 'none',
              background: loading || !input.trim() ? '#1c1c1c' : '#2563eb',
              color: loading || !input.trim() ? '#374151' : '#fff',
              cursor: loading || !input.trim() ? 'default' : 'pointer',
              fontSize: 16,
              transition: 'all 0.1s',
              flexShrink: 0,
            }}
          >↑</button>
        </div>
      </div>
    </PageShell>
  )
}
