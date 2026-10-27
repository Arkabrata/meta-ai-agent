import { useCallback, useEffect, useMemo, useRef, useState, type FormEvent } from 'react'
import { api } from '../api'
import { useAuth } from '../auth'

type Account = {
  id: string
  account_id: string
  name: string
  currency: string
  label: string
}

function todayISO() {
  return new Date().toISOString().slice(0, 10)
}

function daysAgoISO(n: number) {
  const d = new Date()
  d.setDate(d.getDate() - n)
  return d.toISOString().slice(0, 10)
}

export function MetaConnect() {
  const { refreshSession } = useAuth()
  const [displayName, setDisplayName] = useState('')
  const [token, setTokenValue] = useState('')
  const [accounts, setAccounts] = useState<Account[]>([])
  const [accountId, setAccountId] = useState('')
  const [dateStart, setDateStart] = useState(daysAgoISO(29))
  const [dateEnd, setDateEnd] = useState(todayISO())
  const [autoRefresh, setAutoRefresh] = useState(true)
  const [daily, setDaily] = useState(true)
  const [error, setError] = useState('')
  const [ok, setOk] = useState('')
  const [busy, setBusy] = useState(false)
  const [preview, setPreview] = useState<Record<string, unknown>[]>([])
  const [lastSynced, setLastSynced] = useState<string | null>(null)
  const connectedRef = useRef(false)
  const fetchSeq = useRef(0)

  const selectedLabel = useMemo(
    () => accounts.find((a) => a.account_id === accountId || a.id === accountId)?.label,
    [accounts, accountId],
  )

  const fetchInsights = useCallback(
    async (reason: string) => {
      if (!accountId) return
      const seq = ++fetchSeq.current
      setBusy(true)
      setError('')
      try {
        const res = await api.metaFetch(accountId, dateStart, dateEnd, daily)
        if (seq !== fetchSeq.current) return
        setPreview(res.preview)
        setLastSynced(new Date().toLocaleTimeString())
        setOk(`Data flowing · ${res.row_count} rows from ${res.account_label} (${reason})`)
        await refreshSession()
      } catch (err) {
        if (seq !== fetchSeq.current) return
        setError(err instanceof Error ? err.message : 'Fetch failed')
      } finally {
        if (seq === fetchSeq.current) setBusy(false)
      }
    },
    [accountId, dateStart, dateEnd, daily, refreshSession],
  )

  async function connect(e: FormEvent) {
    e.preventDefault()
    setBusy(true)
    setError('')
    setOk('')
    connectedRef.current = false
    try {
      const res = await api.metaConnect(token, displayName)
      setAccounts(res.accounts)
      const firstId = res.accounts[0] ? res.accounts[0].account_id || res.accounts[0].id : ''
      setAccountId(firstId)
      connectedRef.current = Boolean(firstId)
      setOk(
        firstId
          ? `Connected as ${res.user.name} · ${res.accounts.length} ad account(s). Loading insights…`
          : `Connected as ${res.user.name}, but no ad accounts were returned.`,
      )
      await refreshSession()
      if (!firstId) setBusy(false)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Connect failed')
      setBusy(false)
    }
  }

  // Auto-fetch when account or date range changes after a successful connect
  useEffect(() => {
    if (!connectedRef.current || !accountId) return
    const t = window.setTimeout(() => {
      void fetchInsights('auto')
    }, 250)
    return () => window.clearTimeout(t)
  }, [accountId, dateStart, dateEnd, daily, fetchInsights])

  // Keep data flowing while this page is open
  useEffect(() => {
    if (!autoRefresh || !connectedRef.current || !accountId) return
    const id = window.setInterval(() => {
      void fetchInsights('refresh')
    }, 5 * 60 * 1000)
    return () => window.clearInterval(id)
  }, [autoRefresh, accountId, fetchInsights])

  return (
    <>
      <div className="page-head">
        <div>
          <h1>Connect Meta Reporting API</h1>
          <p>
            Facebook username/password cannot be used for the Marketing API (Meta blocks that).
            Paste a Graph API access token, pick an ad account — insights load automatically into
            Dashboard and Chat.
          </p>
        </div>
      </div>

      <div className="panel" style={{ marginBottom: 14 }}>
        <h3>How to connect</h3>
        <ol style={{ margin: 0, paddingLeft: 18, color: 'var(--muted)', fontSize: 13.5, lineHeight: 1.55 }}>
          <li>Create/open an app at developers.facebook.com and add Marketing API.</li>
          <li>Generate a User or System User token with <code>ads_read</code>.</li>
          <li>Paste the token below → accounts load → select one → data syncs on its own.</li>
        </ol>
      </div>

      <form className="panel stack" onSubmit={connect}>
        <h3>1. Access token</h3>
        <div className="field">
          <label>Connection name (optional)</label>
          <input
            value={displayName}
            onChange={(e) => setDisplayName(e.target.value)}
            placeholder="e.g. Media buyer workspace"
          />
        </div>
        <div className="field">
          <label>Meta access token</label>
          <input
            type="password"
            value={token}
            onChange={(e) => setTokenValue(e.target.value)}
            required
            placeholder="EAAB…"
          />
        </div>
        <button className="btn btn-primary" disabled={busy && !accounts.length}>
          {busy && !accounts.length ? 'Connecting…' : 'Connect & load ad accounts'}
        </button>
      </form>

      {accounts.length > 0 && (
        <div className="panel stack">
          <h3>2. Ad account — data flows automatically</h3>
          <div className="field">
            <label>Ad account</label>
            <select
              value={accountId}
              onChange={(e) => {
                connectedRef.current = true
                setAccountId(e.target.value)
              }}
            >
              {accounts.map((a) => (
                <option key={a.id} value={a.account_id || a.id}>
                  {a.label}
                </option>
              ))}
            </select>
          </div>
          <div className="grid-2">
            <div className="field">
              <label>Date start</label>
              <input
                type="date"
                value={dateStart}
                onChange={(e) => {
                  connectedRef.current = true
                  setDateStart(e.target.value)
                }}
              />
            </div>
            <div className="field">
              <label>Date end</label>
              <input
                type="date"
                value={dateEnd}
                onChange={(e) => {
                  connectedRef.current = true
                  setDateEnd(e.target.value)
                }}
              />
            </div>
          </div>
          <label className="row" style={{ fontSize: 13, color: 'var(--muted)' }}>
            <input
              type="checkbox"
              checked={daily}
              onChange={(e) => {
                connectedRef.current = true
                setDaily(e.target.checked)
              }}
            />
            Daily breakdown (one row per ad per day — enables trend charts)
          </label>
          <label className="row" style={{ fontSize: 13, color: 'var(--muted)' }}>
            <input
              type="checkbox"
              checked={autoRefresh}
              onChange={(e) => setAutoRefresh(e.target.checked)}
            />
            Auto-refresh every 5 minutes while this page is open
          </label>
          <div className="row">
            <button
              className="btn btn-ghost"
              disabled={busy}
              onClick={() => fetchInsights('manual')}
              type="button"
            >
              {busy ? 'Syncing…' : 'Sync now'}
            </button>
            {lastSynced && (
              <span style={{ fontSize: 12.5, color: 'var(--muted)' }}>
                Last sync {lastSynced}
                {selectedLabel ? ` · ${selectedLabel}` : ''}
              </span>
            )}
          </div>
        </div>
      )}

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

      {preview.length > 0 && (
        <div className="panel" style={{ marginTop: 16 }}>
          <h3>Live preview</h3>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  {Object.keys(preview[0]).map((k) => (
                    <th key={k}>{k}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {preview.slice(0, 15).map((row, i) => (
                  <tr key={i}>
                    {Object.keys(preview[0]).map((k) => (
                      <td key={k}>{String(row[k] ?? '')}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </>
  )
}
