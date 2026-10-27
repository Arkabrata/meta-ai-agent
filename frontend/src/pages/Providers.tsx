import { useEffect, useMemo, useState, type FormEvent } from 'react'
import { api } from '../api'
import { useAuth } from '../auth'

type Usage = {
  requests: number
  prompt_tokens: number
  completion_tokens: number
  total_tokens: number
  estimated_cost_usd: number
  by_provider: Record<
    string,
    {
      requests: number
      prompt_tokens: number
      completion_tokens: number
      total_tokens: number
      estimated_cost_usd: number
      models: Record<string, number>
    }
  >
  recent: Array<{
    feature: string
    provider: string
    model: string
    prompt_tokens: number
    completion_tokens: number
    total_tokens: number
    estimated_cost_usd: number
  }>
}

const emptyUsage: Usage = {
  requests: 0,
  prompt_tokens: 0,
  completion_tokens: 0,
  total_tokens: 0,
  estimated_cost_usd: 0,
  by_provider: {},
  recent: [],
}

function formatTokens(n: number) {
  return n.toLocaleString()
}

function formatUsd(n: number) {
  if (n < 0.01 && n > 0) return `$${n.toFixed(4)}`
  return `$${n.toFixed(2)}`
}

export function Providers() {
  const { refreshSession } = useAuth()
  const [options, setOptions] = useState<{ id: string; label: string }[]>([])
  const [defaults, setDefaults] = useState<Record<string, string>>({})
  const [models, setModels] = useState<Record<string, string[]>>({})
  const [configured, setConfigured] = useState<
    Array<{ provider: string; label: string; model: string; masked_key: string; active: boolean }>
  >([])
  const [usage, setUsage] = useState<Usage>(emptyUsage)
  const [provider, setProvider] = useState('openai')
  const [label, setLabel] = useState('')
  const [apiKey, setApiKey] = useState('')
  const [model, setModel] = useState('')
  const [customModel, setCustomModel] = useState(false)
  const [baseUrl, setBaseUrl] = useState('')
  const [error, setError] = useState('')
  const [ok, setOk] = useState('')
  const [busy, setBusy] = useState(false)

  const modelOptions = useMemo(() => models[provider] || [], [models, provider])

  async function load() {
    const res = await api.providers()
    setOptions(res.options)
    setDefaults(res.defaults)
    setModels(res.models || {})
    setConfigured(res.configured)
    setUsage(res.usage || emptyUsage)
    const list = res.models?.[provider] || []
    const next = res.defaults[provider] || list[0] || ''
    if (!model || !list.includes(model)) {
      setModel(next)
      setCustomModel(false)
    }
  }

  useEffect(() => {
    load().catch((err) => setError(err instanceof Error ? err.message : 'Failed to load providers'))
  }, [])

  useEffect(() => {
    const list = models[provider] || []
    const next = defaults[provider] || list[0] || ''
    setModel(next)
    setCustomModel(false)
  }, [provider, defaults, models])

  async function save(e: FormEvent) {
    e.preventDefault()
    setBusy(true)
    setError('')
    setOk('')
    try {
      await api.saveProvider({
        provider,
        api_key: apiKey,
        model: model.trim() || defaults[provider] || '',
        base_url: baseUrl,
        label: label || options.find((o) => o.id === provider)?.label || provider,
      })
      setOk('Provider saved and set active')
      setApiKey('')
      await load()
      await refreshSession()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Save failed')
    } finally {
      setBusy(false)
    }
  }

  async function activate(id: string) {
    await api.setActiveProvider(id)
    await load()
    await refreshSession()
  }

  async function remove(id: string, name: string) {
    if (!window.confirm(`Delete provider “${name}”?`)) return
    setBusy(true)
    setError('')
    setOk('')
    try {
      await api.deleteProvider(id)
      setOk(`Deleted ${name}`)
      await load()
      await refreshSession()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Delete failed')
    } finally {
      setBusy(false)
    }
  }

  async function resetUsage() {
    if (!window.confirm('Reset session usage meter?')) return
    const res = await api.resetUsage()
    setUsage(res as Usage)
  }

  return (
    <>
      <div className="page-head">
        <div>
          <h1>AI Providers</h1>
          <p>
            Add API keys for ChatGPT, Claude, Grok, Gemini, or any OpenAI-compatible endpoint. Choose
            which one powers dashboard insights and strategy chat.
          </p>
        </div>
      </div>

      <div className="panel" style={{ marginBottom: 14 }}>
        <div className="row" style={{ justifyContent: 'space-between', marginBottom: 12 }}>
          <h3 style={{ margin: 0 }}>Session API usage</h3>
          <button className="btn btn-ghost" type="button" onClick={resetUsage} disabled={!usage.requests}>
            Reset meter
          </button>
        </div>
        <div className="grid-kpis" style={{ gridTemplateColumns: 'repeat(4, minmax(0, 1fr))' }}>
          <div className="kpi">
            <span>Requests</span>
            <strong>{formatTokens(usage.requests)}</strong>
          </div>
          <div className="kpi">
            <span>Total tokens</span>
            <strong>{formatTokens(usage.total_tokens)}</strong>
          </div>
          <div className="kpi">
            <span>Prompt / completion</span>
            <strong style={{ fontSize: 15 }}>
              {formatTokens(usage.prompt_tokens)} / {formatTokens(usage.completion_tokens)}
            </strong>
          </div>
          <div className="kpi">
            <span>Est. cost (USD)</span>
            <strong>{formatUsd(usage.estimated_cost_usd)}</strong>
          </div>
        </div>
        {Object.keys(usage.by_provider).length > 0 && (
          <div style={{ marginTop: 14 }}>
            <div style={{ fontSize: 12.5, color: 'var(--muted)', marginBottom: 8 }}>By provider</div>
            <div className="stack">
              {Object.entries(usage.by_provider).map(([id, u]) => (
                <div key={id} className="row" style={{ justifyContent: 'space-between', fontSize: 13 }}>
                  <strong>{id}</strong>
                  <span style={{ color: 'var(--muted)' }}>
                    {u.requests} req · {formatTokens(u.total_tokens)} tok · {formatUsd(u.estimated_cost_usd)}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}
        {usage.recent.length > 0 && (
          <div style={{ marginTop: 14 }}>
            <div style={{ fontSize: 12.5, color: 'var(--muted)', marginBottom: 8 }}>Recent calls</div>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Feature</th>
                    <th>Provider</th>
                    <th>Model</th>
                    <th>Tokens</th>
                    <th>Est. $</th>
                  </tr>
                </thead>
                <tbody>
                  {usage.recent.slice(0, 8).map((r, i) => (
                    <tr key={i}>
                      <td>{r.feature || '—'}</td>
                      <td>{r.provider}</td>
                      <td>{r.model}</td>
                      <td>{formatTokens(r.total_tokens)}</td>
                      <td>{formatUsd(r.estimated_cost_usd)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
        {!usage.requests && (
          <p style={{ color: 'var(--muted)', margin: '12px 0 0', fontSize: 13 }}>
            Usage meters as you run chat, chart insights, detailed analysis, or PDF export in this session.
          </p>
        )}
      </div>

      <form className="panel stack" onSubmit={save}>
        <div className="field">
          <label>Provider</label>
          <select value={provider} onChange={(e) => setProvider(e.target.value)}>
            {options.map((o) => (
              <option key={o.id} value={o.id}>
                {o.label}
              </option>
            ))}
          </select>
        </div>
        <div className="field">
          <label>Label</label>
          <input value={label} onChange={(e) => setLabel(e.target.value)} placeholder="Optional display name" />
        </div>
        <div className="field">
          <label>API key</label>
          <input type="password" value={apiKey} onChange={(e) => setApiKey(e.target.value)} required />
        </div>
        <div className="field">
          <label>Model</label>
          {customModel || provider === 'custom' ? (
            <input
              value={model}
              onChange={(e) => setModel(e.target.value)}
              placeholder="Enter model id"
            />
          ) : (
            <select
              value={modelOptions.includes(model) ? model : modelOptions[0] || ''}
              onChange={(e) => {
                if (e.target.value === '__custom__') {
                  setCustomModel(true)
                  setModel('')
                } else {
                  setModel(e.target.value)
                }
              }}
            >
              {modelOptions.map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
              <option value="__custom__">Custom model id…</option>
            </select>
          )}
          {customModel && provider !== 'custom' && (
            <button
              type="button"
              className="btn btn-ghost"
              style={{ marginTop: 8 }}
              onClick={() => {
                setCustomModel(false)
                setModel(defaults[provider] || modelOptions[0] || '')
              }}
            >
              Back to model list
            </button>
          )}
        </div>
        {provider === 'custom' && (
          <div className="field">
            <label>Base URL</label>
            <input
              value={baseUrl}
              onChange={(e) => setBaseUrl(e.target.value)}
              placeholder="https://api.example.com/v1"
            />
          </div>
        )}
        <button className="btn btn-primary" disabled={busy}>
          {busy ? 'Saving…' : 'Save provider'}
        </button>
      </form>

      {error && (
        <div className="alert alert-error" style={{ marginTop: 16 }}>
          {error}
        </div>
      )}
      {ok && (
        <div className="alert alert-ok" style={{ marginTop: 16 }}>
          {ok}
        </div>
      )}

      <div className="panel" style={{ marginTop: 16 }}>
        <h3>Configured providers</h3>
        {configured.length === 0 ? (
          <p style={{ color: 'var(--muted)' }}>No providers yet. Add one above, or set keys in `.env`.</p>
        ) : (
          <div className="stack">
            {configured.map((c) => (
              <div key={c.provider} className="row" style={{ justifyContent: 'space-between' }}>
                <div>
                  <strong>
                    {c.active ? '● ' : '○ '}
                    {c.label}
                  </strong>
                  <div style={{ color: 'var(--muted)', fontSize: '0.85rem' }}>
                    {c.model} · key {c.masked_key}
                  </div>
                </div>
                <div className="row">
                  {!c.active && (
                    <button
                      className="btn btn-ghost"
                      type="button"
                      disabled={busy}
                      onClick={() => activate(c.provider)}
                    >
                      Set active
                    </button>
                  )}
                  <button
                    className="btn btn-ghost"
                    type="button"
                    disabled={busy}
                    style={{ color: 'var(--danger)', borderColor: 'rgba(185, 28, 28, 0.25)' }}
                    onClick={() => remove(c.provider, c.label)}
                  >
                    Delete
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </>
  )
}
