import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { api } from '../api'
import { useAuth } from '../auth'

export function Home() {
  const navigate = useNavigate()
  const { refreshSession } = useAuth()
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [ok, setOk] = useState('')

  async function loadDemo() {
    setBusy(true)
    setError('')
    setOk('')
    try {
      const res = await api.loadSample()
      setOk(`Loaded ${res.row_count} demo rows (SQLite file store)`)
      await refreshSession()
      navigate('/dashboard')
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not load demo data')
    } finally {
      setBusy(false)
    }
  }

  return (
    <>
      <section className="hero">
        <h1>Meta Analyst</h1>
        <p>
          Turn Meta Ads exports and live Reporting API data into KPI dashboards, AI insights, and
          client-ready reports.
        </p>
      </section>

      <div className="feature-grid">
        <div className="feature">
          <h3>Connect</h3>
          <p>Pull live insights via Meta Marketing API, or upload CSV/Excel from Ads Manager.</p>
        </div>
        <div className="feature">
          <h3>Analyze</h3>
          <p>Auto-clean data, compute ROAS/CPC/CTR, and visualize performance instantly.</p>
        </div>
        <div className="feature">
          <h3>Act</h3>
          <p>Chat beside your dashboard with OpenAI, Claude, Grok, or Gemini and export PDFs.</p>
        </div>
      </div>

      <div className="panel" style={{ marginTop: 14 }}>
        <h3>Get started</h3>
        <p style={{ color: 'var(--muted)', margin: '0 0 14px', fontSize: 13.5 }}>
          Load 150 generic demo ads into the local SQLite file (no database server), or connect /
          upload your own Meta report.
        </p>
        <div className="row">
          <button className="btn btn-primary" type="button" disabled={busy} onClick={loadDemo}>
            {busy ? 'Loading…' : 'Load demo data (150 rows)'}
          </button>
          <Link className="btn btn-ghost" to="/upload">
            Upload report
          </Link>
          <Link className="btn btn-ghost" to="/meta">
            Connect Meta API
          </Link>
        </div>
        {error && <div className="alert alert-error" style={{ marginTop: 12 }}>{error}</div>}
        {ok && <div className="alert alert-ok" style={{ marginTop: 12 }}>{ok}</div>}
      </div>
    </>
  )
}
