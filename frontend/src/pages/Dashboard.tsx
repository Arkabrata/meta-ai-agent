import { useEffect, useMemo, useState, type ReactNode } from 'react'
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ComposedChart,
  Legend,
  Line,
  Pie,
  PieChart,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from 'recharts'
import { api } from '../api'
import { ChatPanel } from './Chat'

const COLORS = ['#0f766e', '#d97706', '#164e63', '#b45309', '#0e7490', '#a16207', '#115e59', '#c2410c']

type Dash = Awaited<ReturnType<typeof api.dashboard>>
type Priority = { priority: string; title: string; details: string }

function money(n: number, currency: string) {
  return `${currency}${Math.round(n).toLocaleString()}`
}

function num(n: number, digits = 0) {
  return n.toLocaleString(undefined, {
    maximumFractionDigits: digits,
    minimumFractionDigits: digits,
  })
}

export function Dashboard() {
  const [data, setData] = useState<Dash | null>(null)
  const [insights, setInsights] = useState<Record<string, string | null>>({})
  const [error, setError] = useState('')
  const [busyInsight, setBusyInsight] = useState<string | null>(null)
  const [detailed, setDetailed] = useState<{ overall_summary: string; priority_blocks: Priority[] } | null>(null)
  const [rawDetailed, setRawDetailed] = useState('')
  const [busy, setBusy] = useState(false)

  async function load() {
    setError('')
    try {
      const res = await api.dashboard()
      setData(res)
      setInsights(res.insights || {})
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load dashboard')
    }
  }

  useEffect(() => {
    load()
  }, [])

  async function loadInsight(key: string) {
    setBusyInsight(key)
    try {
      const res = await api.chartInsight(key)
      setInsights((prev) => ({ ...prev, [key]: res.insight }))
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Insight failed')
    } finally {
      setBusyInsight(null)
    }
  }

  async function runDetailed() {
    setBusy(true)
    setError('')
    try {
      const res = await api.detailedAnalysis()
      setDetailed(res.parsed)
      setRawDetailed(res.raw)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Analysis failed')
    } finally {
      setBusy(false)
    }
  }

  async function downloadPdf() {
    setBusy(true)
    try {
      await api.downloadPdf()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'PDF failed')
    } finally {
      setBusy(false)
    }
  }

  const currency = data?.currency || '₹'
  const moneyTip = useMemo(
    () =>
      ({ active, payload, label }: any) => {
        if (!active || !payload?.length) return null
        return (
          <div className="chart-tooltip">
            <div className="chart-tooltip-label">{payload[0]?.payload?.full_name || label}</div>
            {payload.map((p: any) => (
              <div key={p.dataKey}>
                {p.name}:{' '}
                <strong>
                  {p.dataKey === 'roas' || p.dataKey === 'ctr' || p.name === 'ROAS'
                    ? num(Number(p.value), 2)
                    : p.dataKey === 'pct'
                      ? `${num(Number(p.value), 1)}%`
                      : typeof p.value === 'number' && (p.dataKey === 'spend' || p.dataKey === 'revenue' || p.dataKey === 'value')
                        ? money(Number(p.value), currency)
                        : num(Number(p.value), 2)}
                </strong>
              </div>
            ))}
          </div>
        )
      },
    [currency],
  )

  if (error && !data) {
    return (
      <>
        <div className="page-head">
          <h1>Performance Dashboard</h1>
        </div>
        <div className="alert alert-error">{error}</div>
      </>
    )
  }

  if (!data) return <div className="panel">Loading dashboard…</div>

  const priorityColor: Record<string, string> = {
    HIGH: '#b91c1c',
    MEDIUM: '#d97706',
    LOW: '#047857',
  }
  const c = data.charts

  return (
    <>
      <div className="page-head">
        <div>
          <h1>Performance Dashboard</h1>
          <p>
            {data.row_count.toLocaleString()} ads
            {data.summary ? ` · ${data.summary.campaigns} campaigns · ${data.summary.objectives} objectives` : ''}
            {' · '}
            chat on the right
          </p>
        </div>
        <div className="row">
          <button className="btn btn-ghost" disabled={busy} onClick={downloadPdf}>
            Download PDF
          </button>
          <button className="btn btn-primary" disabled={busy} onClick={runDetailed}>
            {busy ? 'Working…' : 'Generate AI Analysis'}
          </button>
        </div>
      </div>

      {error && (
        <div className="alert alert-error" style={{ marginBottom: 16 }}>
          {error}
        </div>
      )}

      <div className="dash-layout">
        <div className="dash-main">
          <div className="grid-kpis">
            {data.kpis.map((k) => (
              <div className="kpi" key={k.name}>
                <span>{k.name}</span>
                <strong>{k.value}</strong>
              </div>
            ))}
          </div>

          {c.trend && (
            <ChartPanel
              title="Performance over time"
              subtitle="Daily spend & revenue (bars) with ROAS (line)"
            >
              <ResponsiveContainer width="100%" height={280}>
                <ComposedChart data={c.trend} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e8e4dc" />
                  <XAxis
                    dataKey="date"
                    tick={{ fontSize: 10 }}
                    tickFormatter={(d: string) => d.slice(5)}
                    minTickGap={18}
                  />
                  <YAxis yAxisId="left" tick={{ fontSize: 11 }} tickFormatter={(v) => `${Math.round(v / 1000)}k`} />
                  <YAxis yAxisId="right" orientation="right" tick={{ fontSize: 11 }} domain={[0, 'auto']} />
                  <Tooltip content={moneyTip} />
                  <Legend />
                  <Bar yAxisId="left" dataKey="spend" name="Spend" fill="#0f766e" radius={[3, 3, 0, 0]} />
                  <Bar yAxisId="left" dataKey="revenue" name="Revenue" fill="#f0b35a" radius={[3, 3, 0, 0]} />
                  <Line yAxisId="right" type="monotone" dataKey="roas" name="ROAS" stroke="#164e63" strokeWidth={2} dot={false} />
                </ComposedChart>
              </ResponsiveContainer>
            </ChartPanel>
          )}

          <div className="grid-2" style={{ marginTop: 14 }}>
            {c.objective_spend && (
              <ChartPanel
                title="Spend by objective"
                subtitle="Share of total media spend"
                chartKey="objective_spend"
                insight={insights.objective_spend}
                busy={busyInsight === 'objective_spend'}
                onInsight={() => loadInsight('objective_spend')}
              >
                <ResponsiveContainer width="100%" height={280}>
                  <PieChart>
                    <Pie
                      data={c.objective_spend}
                      dataKey="value"
                      nameKey="name"
                      innerRadius={58}
                      outerRadius={92}
                      paddingAngle={2}
                    >
                      {c.objective_spend.map((_, i) => (
                        <Cell key={i} fill={COLORS[i % COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip content={moneyTip} />
                    <Legend
                      verticalAlign="bottom"
                      formatter={(value, entry: any) =>
                        `${value} (${entry?.payload?.pct ?? 0}%)`
                      }
                    />
                  </PieChart>
                </ResponsiveContainer>
              </ChartPanel>
            )}

            {c.objective_spend_revenue && (
              <ChartPanel
                title="Spend vs revenue by objective"
                subtitle="Bars = spend / revenue · line = ROAS"
                chartKey="objective_roas"
                insight={insights.objective_roas}
                busy={busyInsight === 'objective_roas'}
                onInsight={() => loadInsight('objective_roas')}
              >
                <ResponsiveContainer width="100%" height={280}>
                  <ComposedChart data={c.objective_spend_revenue} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e8e4dc" />
                    <XAxis dataKey="name" tick={{ fontSize: 11 }} />
                    <YAxis yAxisId="left" tick={{ fontSize: 11 }} tickFormatter={(v) => `${Math.round(v / 1000)}k`} />
                    <YAxis yAxisId="right" orientation="right" tick={{ fontSize: 11 }} domain={[0, 'auto']} />
                    <Tooltip content={moneyTip} />
                    <Legend />
                    <Bar yAxisId="left" dataKey="spend" name="Spend" fill="#0f766e" radius={[4, 4, 0, 0]} />
                    <Bar yAxisId="left" dataKey="revenue" name="Revenue" fill="#d97706" radius={[4, 4, 0, 0]} />
                    <Line yAxisId="right" type="monotone" dataKey="roas" name="ROAS" stroke="#164e63" strokeWidth={2} dot />
                  </ComposedChart>
                </ResponsiveContainer>
              </ChartPanel>
            )}

            {c.campaign_spend && (
              <ChartPanel
                title="Top campaigns by spend"
                subtitle="Spend with revenue overlay"
              >
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={c.campaign_spend} layout="vertical" margin={{ left: 8, right: 12 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e8e4dc" />
                    <XAxis type="number" tick={{ fontSize: 11 }} tickFormatter={(v) => `${Math.round(v / 1000)}k`} />
                    <YAxis type="category" dataKey="name" width={110} tick={{ fontSize: 11 }} />
                    <Tooltip content={moneyTip} />
                    <Legend />
                    <Bar dataKey="spend" name="Spend" fill="#0f766e" radius={[0, 4, 4, 0]} />
                    <Bar dataKey="revenue" name="Revenue" fill="#f0b35a" radius={[0, 4, 4, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </ChartPanel>
            )}

            {c.funnel && (
              <ChartPanel title="Conversion funnel" subtitle="% of top-of-funnel impressions">
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={c.funnel} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e8e4dc" />
                    <XAxis dataKey="name" tick={{ fontSize: 11 }} />
                    <YAxis tick={{ fontSize: 11 }} tickFormatter={(v) => (v >= 1000 ? `${Math.round(v / 1000)}k` : String(v))} />
                    <Tooltip
                      formatter={(value: any, name: any) => {
                        if (name === 'value' || name === 'Count') return [num(Number(value)), 'Count']
                        return [value, name]
                      }}
                      labelFormatter={(_, payload) =>
                        payload?.[0]?.payload
                          ? `${payload[0].payload.name} · ${payload[0].payload.pct}% of impressions`
                          : ''
                      }
                    />
                    <Bar dataKey="value" name="Count" radius={[6, 6, 0, 0]}>
                      {c.funnel.map((_, i) => (
                        <Cell key={i} fill={COLORS[i % COLORS.length]} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
                <div className="funnel-meta">
                  {c.funnel.map((s) => (
                    <span key={s.name}>
                      {s.name}: <strong>{num(s.value)}</strong> ({s.pct}%)
                    </span>
                  ))}
                </div>
              </ChartPanel>
            )}

            {c.scatter_spend_roas && (
              <ChartPanel
                title="Spend vs ROAS efficiency"
                subtitle="Bubble size ≈ purchases · hover for campaign"
                chartKey="scatter_spend_roas"
                insight={insights.scatter_spend_roas}
                busy={busyInsight === 'scatter_spend_roas'}
                onInsight={() => loadInsight('scatter_spend_roas')}
              >
                <ResponsiveContainer width="100%" height={300}>
                  <ScatterChart margin={{ top: 8, right: 12, bottom: 8, left: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e8e4dc" />
                    <XAxis
                      type="number"
                      dataKey="spend"
                      name="Spend"
                      tick={{ fontSize: 11 }}
                      tickFormatter={(v) => `${Math.round(v / 1000)}k`}
                    />
                    <YAxis type="number" dataKey="roas" name="ROAS" tick={{ fontSize: 11 }} />
                    <ZAxis type="number" dataKey="purchases" range={[40, 280]} />
                    <Tooltip
                      cursor={{ strokeDasharray: '3 3' }}
                      formatter={(value: any, name: any) => {
                        if (name === 'Spend') return [money(Number(value), currency), name]
                        return [num(Number(value), 2), name]
                      }}
                      labelFormatter={(_, payload) => payload?.[0]?.payload?.campaign || 'Ad'}
                    />
                    <Scatter data={c.scatter_spend_roas} fill="#d97706" fillOpacity={0.7} />
                  </ScatterChart>
                </ResponsiveContainer>
              </ChartPanel>
            )}

            {c.adset_perf && (
              <ChartPanel title="Ad set performance" subtitle="Top ad sets by spend · ROAS line">
                <ResponsiveContainer width="100%" height={300}>
                  <ComposedChart data={c.adset_perf} margin={{ top: 8, right: 12, left: 0, bottom: 48 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e8e4dc" />
                    <XAxis dataKey="name" tick={{ fontSize: 10 }} angle={-28} textAnchor="end" interval={0} height={60} />
                    <YAxis yAxisId="left" tick={{ fontSize: 11 }} tickFormatter={(v) => `${Math.round(v / 1000)}k`} />
                    <YAxis yAxisId="right" orientation="right" tick={{ fontSize: 11 }} domain={[0, 'auto']} />
                    <Tooltip content={moneyTip} />
                    <Legend verticalAlign="top" />
                    <Bar yAxisId="left" dataKey="spend" name="Spend" fill="#0e7490" radius={[4, 4, 0, 0]} />
                    <Line yAxisId="right" type="monotone" dataKey="roas" name="ROAS" stroke="#d97706" strokeWidth={2} dot />
                  </ComposedChart>
                </ResponsiveContainer>
              </ChartPanel>
            )}

            {c.roas_distribution && (
              <ChartPanel title="ROAS distribution" subtitle="How spend is spread across ROAS bands">
                <ResponsiveContainer width="100%" height={300}>
                  <ComposedChart data={c.roas_distribution} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e8e4dc" />
                    <XAxis dataKey="bucket" tick={{ fontSize: 11 }} />
                    <YAxis yAxisId="left" tick={{ fontSize: 11 }} tickFormatter={(v) => `${Math.round(v / 1000)}k`} />
                    <YAxis yAxisId="right" orientation="right" tick={{ fontSize: 11 }} allowDecimals={false} />
                    <Tooltip
                      formatter={(value: any, name: any) =>
                        name === 'Spend' ? [money(Number(value), currency), name] : [num(Number(value)), name]
                      }
                      labelFormatter={(l) => `ROAS ${l}`}
                    />
                    <Legend verticalAlign="top" />
                    <Bar yAxisId="left" dataKey="spend" name="Spend" radius={[4, 4, 0, 0]}>
                      {c.roas_distribution.map((_, i) => (
                        <Cell
                          key={i}
                          fill={i < 2 ? '#b91c1c' : i < 4 ? '#d97706' : '#0f766e'}
                          fillOpacity={0.85}
                        />
                      ))}
                    </Bar>
                    <Line yAxisId="right" type="monotone" dataKey="ads" name="Ads" stroke="#164e63" strokeWidth={2} dot />
                  </ComposedChart>
                </ResponsiveContainer>
              </ChartPanel>
            )}

            {c.cost_efficiency && (
              <ChartPanel title="Cost efficiency" subtitle="CPC and CPM by campaign (lower is better)">
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={c.cost_efficiency} margin={{ top: 8, right: 12, left: 0, bottom: 48 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e8e4dc" />
                    <XAxis dataKey="name" tick={{ fontSize: 10 }} angle={-28} textAnchor="end" interval={0} height={60} />
                    <YAxis tick={{ fontSize: 11 }} />
                    <Tooltip
                      formatter={(value: any, name: any) => [money(Number(value), currency), name]}
                      labelFormatter={(_, payload) => payload?.[0]?.payload?.full_name || ''}
                    />
                    <Legend verticalAlign="top" />
                    <Bar dataKey="cpc" name="CPC" fill="#0f766e" radius={[4, 4, 0, 0]} />
                    <Bar dataKey="cpm" name="CPM" fill="#a16207" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </ChartPanel>
            )}

            {c.reach_frequency && (
              <ChartPanel title="Reach & frequency" subtitle="Unique reach (bars) and avg frequency (line)">
                <ResponsiveContainer width="100%" height={300}>
                  <ComposedChart data={c.reach_frequency} margin={{ top: 8, right: 12, left: 0, bottom: 48 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e8e4dc" />
                    <XAxis dataKey="name" tick={{ fontSize: 10 }} angle={-28} textAnchor="end" interval={0} height={60} />
                    <YAxis yAxisId="left" tick={{ fontSize: 11 }} tickFormatter={(v) => `${Math.round(v / 1000)}k`} />
                    <YAxis yAxisId="right" orientation="right" tick={{ fontSize: 11 }} domain={[0, 'auto']} />
                    <Tooltip
                      formatter={(value: any, name: any) =>
                        name === 'Reach' ? [num(Number(value)), name] : [num(Number(value), 2), name]
                      }
                      labelFormatter={(_, payload) => payload?.[0]?.payload?.full_name || ''}
                    />
                    <Legend verticalAlign="top" />
                    <Bar yAxisId="left" dataKey="reach" name="Reach" fill="#164e63" radius={[4, 4, 0, 0]} />
                    <Line yAxisId="right" type="monotone" dataKey="frequency" name="Frequency" stroke="#d97706" strokeWidth={2} dot />
                  </ComposedChart>
                </ResponsiveContainer>
              </ChartPanel>
            )}

            {c.ctr_by_campaign && (
              <ChartPanel
                title="CTR by campaign"
                subtitle="Average click-through rate (%)"
                chartKey="ctr_by_campaign"
                insight={insights.ctr_by_campaign}
                busy={busyInsight === 'ctr_by_campaign'}
                onInsight={() => loadInsight('ctr_by_campaign')}
              >
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={c.ctr_by_campaign} margin={{ bottom: 48 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e8e4dc" />
                    <XAxis dataKey="name" tick={{ fontSize: 10 }} angle={-28} textAnchor="end" interval={0} height={60} />
                    <YAxis tick={{ fontSize: 11 }} unit="%" />
                    <Tooltip
                      formatter={(value: any) => [`${num(Number(value), 2)}%`, 'CTR']}
                      labelFormatter={(_, payload) => payload?.[0]?.payload?.full_name || ''}
                    />
                    <Bar dataKey="value" name="CTR" fill="#164e63" radius={[6, 6, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </ChartPanel>
            )}
          </div>

          {c.campaign_table && (
            <div className="panel" style={{ marginTop: 14 }}>
              <h3>Campaign scorecard</h3>
              <p style={{ margin: '0 0 10px', color: 'var(--muted)', fontSize: 13 }}>
                Ranked by spend · ROAS and CTR for quick triage
              </p>
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>Campaign</th>
                      <th>Spend</th>
                      <th>Revenue</th>
                      <th>ROAS</th>
                      <th>CTR</th>
                      <th>CPC</th>
                      <th>Clicks</th>
                      <th>Purchases</th>
                    </tr>
                  </thead>
                  <tbody>
                    {c.campaign_table.map((r) => (
                      <tr key={r.campaign}>
                        <td style={{ whiteSpace: 'normal', minWidth: 160 }}>{r.campaign}</td>
                        <td>{money(r.spend, currency)}</td>
                        <td>{money(r.revenue, currency)}</td>
                        <td>
                          <span className={r.roas >= 2 ? 'tag-good' : r.roas < 1 ? 'tag-bad' : 'tag-mid'}>
                            {num(r.roas, 2)}
                          </span>
                        </td>
                        <td>{num(r.ctr, 2)}%</td>
                        <td>{money(r.cpc, currency)}</td>
                        <td>{num(r.clicks)}</td>
                        <td>{num(r.purchases)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {(c.top_ads || c.weak_ads) && (
            <div className="grid-2" style={{ marginTop: 14 }}>
              {c.top_ads && (
                <div className="panel">
                  <h3>Strongest ads</h3>
                  <p style={{ margin: '0 0 10px', color: 'var(--muted)', fontSize: 13 }}>Highest ROAS</p>
                  <div className="table-wrap">
                    <table>
                      <thead>
                        <tr>
                          <th>Ad</th>
                          <th>Spend</th>
                          <th>ROAS</th>
                          <th>Purch.</th>
                        </tr>
                      </thead>
                      <tbody>
                        {c.top_ads.map((r, i) => (
                          <tr key={`${r.ad}-${i}`}>
                            <td style={{ whiteSpace: 'normal' }}>
                              <div>{r.ad}</div>
                              <div style={{ color: 'var(--muted)', fontSize: 11 }}>{r.campaign}</div>
                            </td>
                            <td>{money(r.spend, currency)}</td>
                            <td className="tag-good">{num(r.roas, 2)}</td>
                            <td>{r.purchases}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
              {c.weak_ads && (
                <div className="panel">
                  <h3>Weak ads (meaningful spend)</h3>
                  <p style={{ margin: '0 0 10px', color: 'var(--muted)', fontSize: 13 }}>
                    Lowest ROAS among higher-spend ads — pause / refresh candidates
                  </p>
                  <div className="table-wrap">
                    <table>
                      <thead>
                        <tr>
                          <th>Ad</th>
                          <th>Spend</th>
                          <th>ROAS</th>
                          <th>Purch.</th>
                        </tr>
                      </thead>
                      <tbody>
                        {c.weak_ads.map((r, i) => (
                          <tr key={`${r.ad}-${i}`}>
                            <td style={{ whiteSpace: 'normal' }}>
                              <div>{r.ad}</div>
                              <div style={{ color: 'var(--muted)', fontSize: 11 }}>{r.campaign}</div>
                            </td>
                            <td>{money(r.spend, currency)}</td>
                            <td className="tag-bad">{num(r.roas, 2)}</td>
                            <td>{r.purchases}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </div>
          )}

          {(detailed || rawDetailed) && (
            <div className="panel" style={{ marginTop: 14 }}>
              <h3>Detailed AI Analysis</h3>
              {detailed ? (
                <>
                  <p>{detailed.overall_summary}</p>
                  {detailed.priority_blocks?.map((b, i) => (
                    <div
                      key={i}
                      className="priority"
                      style={{ borderColor: priorityColor[b.priority] || '#111' }}
                    >
                      <h4 style={{ color: priorityColor[b.priority] || '#111' }}>
                        {b.priority} PRIORITY — {b.title}
                      </h4>
                      <p style={{ margin: 0 }}>{b.details}</p>
                    </div>
                  ))}
                </>
              ) : (
                <pre style={{ whiteSpace: 'pre-wrap' }}>{rawDetailed}</pre>
              )}
            </div>
          )}
        </div>

        <aside className="dash-chat">
          <ChatPanel compact />
        </aside>
      </div>
    </>
  )
}

function ChartPanel({
  title,
  subtitle,
  insight,
  busy,
  onInsight,
  children,
}: {
  title: string
  subtitle?: string
  chartKey?: string
  insight?: string | null
  busy?: boolean
  onInsight?: () => void
  children: ReactNode
}) {
  return (
    <div className="panel">
      <div className="row" style={{ justifyContent: 'space-between', marginBottom: 4 }}>
        <div>
          <h3 style={{ margin: 0 }}>{title}</h3>
          {subtitle && (
            <p style={{ margin: '4px 0 0', color: 'var(--muted)', fontSize: 12.5 }}>{subtitle}</p>
          )}
        </div>
        {onInsight && (
          <button className="btn btn-ghost" type="button" disabled={busy} onClick={onInsight}>
            {busy ? '…' : insight ? 'Refresh insight' : 'AI insight'}
          </button>
        )}
      </div>
      {children}
      {insight && <div className="insight">{insight}</div>}
    </div>
  )
}
