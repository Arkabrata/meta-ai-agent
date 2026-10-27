import { useEffect, useState } from 'react'
import { api } from '../api'
import { useAuth } from '../auth'

type Schema = Awaited<ReturnType<typeof api.reportSchema>>
type Mapping = Awaited<ReturnType<typeof api.upload>>['mapping']

export function Upload() {
  const { refreshSession } = useAuth()
  const [schema, setSchema] = useState<Schema | null>(null)
  const [datasets, setDatasets] = useState<
    Array<{ id: string; name: string; source: string; row_count: number }>
  >([])
  const [storeNote, setStoreNote] = useState('')
  const [error, setError] = useState('')
  const [ok, setOk] = useState('')
  const [busy, setBusy] = useState(false)
  const [preview, setPreview] = useState<Record<string, unknown>[]>([])
  const [mapping, setMapping] = useState<Mapping | null>(null)

  async function refreshDatasets() {
    const [list, info] = await Promise.all([api.datasets(), api.storeInfo()])
    setDatasets(list.datasets)
    setStoreNote(`${info.engine} · ${info.note}`)
  }

  useEffect(() => {
    api
      .reportSchema()
      .then(setSchema)
      .catch((err) => setError(err instanceof Error ? err.message : 'Failed to load schema'))
    refreshDatasets().catch(() => undefined)
  }, [])

  async function onFile(file: File | null) {
    if (!file) return
    setBusy(true)
    setError('')
    setOk('')
    setMapping(null)
    try {
      const res = await api.upload(file)
      setPreview(res.preview)
      setMapping(res.mapping)
      setOk(
        `Processed ${res.row_count} rows · mapped ${res.mapping.coverage.mapped}/${res.mapping.coverage.total_supported} fields · saved to SQLite`,
      )
      await refreshSession()
      await refreshDatasets()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload failed')
    } finally {
      setBusy(false)
    }
  }

  async function loadSample() {
    setBusy(true)
    setError('')
    setOk('')
    try {
      const res = await api.loadSample()
      setPreview(res.preview)
      setMapping(null)
      setOk(`Loaded demo dataset · ${res.row_count} rows from SQLite`)
      await refreshSession()
      await refreshDatasets()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Sample load failed')
    } finally {
      setBusy(false)
    }
  }

  async function loadSaved(id: string) {
    setBusy(true)
    setError('')
    setOk('')
    try {
      const res = await api.loadDataset(id)
      setPreview(res.preview)
      setOk(`Loaded ${res.row_count} rows from saved dataset`)
      await refreshSession()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Load failed')
    } finally {
      setBusy(false)
    }
  }

  return (
    <>
      <div className="page-head">
        <div>
          <h1>Upload Meta Ads Report</h1>
          <p>
            Import a full Meta Ads Manager CSV/Excel export. Data is stored in a local SQLite file
            (no database server). Use the structure below for correct column mapping.
          </p>
        </div>
        <div className="row">
          <button className="btn btn-primary" type="button" disabled={busy} onClick={loadSample}>
            {busy ? '…' : 'Load demo (150 rows)'}
          </button>
          <button
            className="btn btn-ghost"
            type="button"
            onClick={() => api.downloadTemplate().catch((e) => setError(e.message))}
          >
            Download template CSV
          </button>
        </div>
      </div>

      {storeNote && (
        <div className="alert alert-info" style={{ marginBottom: 14 }}>
          Local store: {storeNote}
        </div>
      )}

      {datasets.length > 0 && (
        <div className="panel" style={{ marginBottom: 14 }}>
          <h3>Saved datasets (SQLite)</h3>
          <div className="stack">
            {datasets.map((d) => (
              <div key={d.id} className="row" style={{ justifyContent: 'space-between' }}>
                <div>
                  <strong>{d.name}</strong>
                  <div style={{ color: 'var(--muted)', fontSize: 12.5 }}>
                    {d.source} · {d.row_count} rows · {d.id}
                  </div>
                </div>
                <button className="btn btn-ghost" type="button" disabled={busy} onClick={() => loadSaved(d.id)}>
                  Load
                </button>
              </div>
            ))}
          </div>
        </div>
      )}

      {schema && (
        <div className="panel" style={{ marginBottom: 14 }}>
          <h3>Full Meta report structure</h3>
          <p style={{ color: 'var(--muted)', marginTop: 0, fontSize: 13.5 }}>{schema.description}</p>
          <p style={{ color: 'var(--muted)', marginTop: 6, fontSize: 13 }}>
            This is the same structure the Meta API connection produces, so uploads and live data
            behave identically. Extra columns are kept as-is (shown as “unused” after upload) and
            never break the import. Add a “Day” breakdown in Ads Manager to unlock trend charts.
          </p>
          {Object.entries(schema.groups).map(([group, fields]) => (
            <div key={group} style={{ marginTop: 14 }}>
              <div style={{ fontSize: 12.5, fontWeight: 600, marginBottom: 6 }}>{group}</div>
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>Column in Ads Manager</th>
                      <th>Required</th>
                      <th>Also accepts</th>
                    </tr>
                  </thead>
                  <tbody>
                    {fields.map((f) => (
                      <tr key={f.key}>
                        <td>
                          <strong>{f.label}</strong>
                        </td>
                        <td>{f.required ? 'Yes' : 'Optional'}</td>
                        <td style={{ whiteSpace: 'normal', maxWidth: 360 }}>
                          {f.aliases.filter((a) => a !== f.label).join(', ') || '—'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          ))}
        </div>
      )}

      <div className="panel stack">
        <div className="field">
          <label>Report file (.csv or .xlsx)</label>
          <input
            type="file"
            accept=".csv,.xlsx"
            disabled={busy}
            onChange={(e) => onFile(e.target.files?.[0] || null)}
          />
        </div>
        {busy && <div className="alert alert-info">Processing…</div>}
        {error && <div className="alert alert-error">{error}</div>}
        {ok && <div className="alert alert-ok">{ok}</div>}
      </div>

      {mapping && (
        <div className="panel" style={{ marginTop: 14 }}>
          <h3>Column mapping</h3>
          <p style={{ color: 'var(--muted)', fontSize: 13 }}>
            Mapped {mapping.coverage.mapped} of {mapping.coverage.total_supported} supported fields
            {mapping.missing_optional.length
              ? ` · ${mapping.missing_optional.length} optional missing`
              : ''}
          </p>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Field</th>
                  <th>Group</th>
                  <th>Source column</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {mapping.found.map((f) => (
                  <tr key={f.key}>
                    <td>{f.label}</td>
                    <td>{f.group}</td>
                    <td>{f.source_column}</td>
                    <td style={{ color: 'var(--ok)' }}>Mapped</td>
                  </tr>
                ))}
                {mapping.missing_optional.map((f) => (
                  <tr key={f.key}>
                    <td>{f.label}</td>
                    <td>—</td>
                    <td>—</td>
                    <td style={{ color: 'var(--muted)' }}>Optional missing</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {mapping.unmapped_source_columns.length > 0 && (
            <p style={{ fontSize: 12.5, color: 'var(--muted)', marginBottom: 0 }}>
              Extra columns kept but unused: {mapping.unmapped_source_columns.join(', ')}
            </p>
          )}
        </div>
      )}

      {preview.length > 0 && (
        <div className="panel" style={{ marginTop: 14 }}>
          <h3>Normalized preview</h3>
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
