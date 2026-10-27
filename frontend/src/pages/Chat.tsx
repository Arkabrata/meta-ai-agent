import { useEffect, useRef, useState, type FormEvent } from 'react'
import { api } from '../api'

type Msg = { role: string; content: string }

/** Embedded chat used on the Dashboard (and optional standalone redirect). */
export function ChatPanel({ compact = false }: { compact?: boolean }) {
  const [messages, setMessages] = useState<Msg[]>([])
  const [input, setInput] = useState('')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const endRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    api
      .getChat()
      .then((res) => setMessages(res.messages))
      .catch((err) => setError(err instanceof Error ? err.message : 'Failed to load chat'))
  }, [])

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  async function send(e: FormEvent) {
    e.preventDefault()
    if (!input.trim() || busy) return
    const text = input.trim()
    setInput('')
    setBusy(true)
    setError('')
    setMessages((m) => [...m, { role: 'user', content: text }])
    try {
      const res = await api.sendChat(text)
      setMessages(res.messages)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Chat failed')
    } finally {
      setBusy(false)
    }
  }

  async function clear() {
    await api.clearChat()
    setMessages([])
  }

  return (
    <div className="panel" style={{ display: 'flex', flexDirection: 'column', minHeight: compact ? 420 : 480 }}>
      <div className="row" style={{ justifyContent: 'space-between', marginBottom: 10 }}>
        <div>
          <h3 style={{ margin: 0 }}>AI Strategy Chat</h3>
          {!compact && (
            <p style={{ margin: '4px 0 0', color: 'var(--muted)', fontSize: 13 }}>
              Ask about ROAS, scaling, pauses, or budget shifts.
            </p>
          )}
        </div>
        <button className="btn btn-ghost" type="button" onClick={clear}>
          Clear
        </button>
      </div>

      <div className="chat-log" style={{ flex: 1, maxHeight: compact ? '46vh' : '55vh' }}>
        {messages.length === 0 && (
          <div style={{ color: 'var(--muted)' }}>
            No messages yet. Try “Which campaigns should I pause?”
          </div>
        )}
        {messages.map((m, i) => (
          <div key={i} className={`bubble ${m.role === 'user' ? 'user' : 'assistant'}`}>
            {m.content}
          </div>
        ))}
        <div ref={endRef} />
      </div>

      {error && (
        <div className="alert alert-error" style={{ marginBottom: 12 }}>
          {error}
        </div>
      )}

      <form className="row" onSubmit={send}>
        <input
          style={{
            flex: 1,
            borderRadius: 6,
            border: '1px solid var(--border)',
            padding: '10px 12px',
          }}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask about ROAS, scaling, pauses…"
          disabled={busy}
        />
        <button className="btn btn-primary" disabled={busy || !input.trim()}>
          {busy ? '…' : 'Send'}
        </button>
      </form>
    </div>
  )
}

export function Chat() {
  return (
    <>
      <div className="page-head">
        <div>
          <h1>AI Strategy Chat</h1>
          <p>Chat lives on the Dashboard alongside your KPIs and charts.</p>
        </div>
      </div>
      <ChatPanel />
    </>
  )
}
