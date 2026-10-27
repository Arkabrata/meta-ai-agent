const TOKEN_KEY = 'meta_analyst_token'

export function getToken(): string | null {
  return localStorage.getItem(TOKEN_KEY)
}

export function setToken(token: string | null) {
  if (token) localStorage.setItem(TOKEN_KEY, token)
  else localStorage.removeItem(TOKEN_KEY)
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers || {})
  const token = getToken()
  if (token) headers.set('Authorization', `Bearer ${token}`)
  if (init.body && !(init.body instanceof FormData) && !headers.has('Content-Type')) {
    headers.set('Content-Type', 'application/json')
  }
  const res = await fetch(path, { ...init, headers })
  if (res.status === 401) {
    setToken(null)
    throw new Error('Unauthorized')
  }
  if (!res.ok) {
    let detail = res.statusText
    try {
      const body = await res.json()
      detail = body.detail || JSON.stringify(body)
    } catch {
      /* ignore */
    }
    throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail))
  }
  if (res.status === 204) return undefined as T
  const ct = res.headers.get('content-type') || ''
  if (ct.includes('application/json')) return res.json()
  return res as unknown as T
}

export const api = {
  brand: () => request<{ product: string; company: string; currency: string }>('/api/brand'),
  login: (password: string) =>
    request<{ token: string; product: string; company: string }>('/api/auth/login', {
      method: 'POST',
      body: JSON.stringify({ password }),
    }),
  logout: () => request<{ ok: boolean }>('/api/auth/logout', { method: 'POST' }),
  session: () =>
    request<{
      has_data: boolean
      row_count: number
      meta: {
        connected: boolean
        display_name?: string
        user_name?: string
        account_label?: string
        account_id?: string
      }
      provider: { id: string; label: string; model: string; has_key: boolean } | null
    }>('/api/session'),
  loadSample: () =>
    request<{
      dataset_id: string
      row_count: number
      columns: string[]
      preview: Record<string, unknown>[]
      store: { engine: string; path: string; note: string; datasets: unknown[] }
    }>('/api/data/sample', { method: 'POST' }),
  datasets: () =>
    request<{
      datasets: Array<{
        id: string
        name: string
        source: string
        row_count: number
        created_at: string
      }>
    }>('/api/data/datasets'),
  loadDataset: (dataset_id: string) =>
    request<{ dataset_id: string; row_count: number; preview: Record<string, unknown>[] }>(
      '/api/data/load',
      {
        method: 'POST',
        body: JSON.stringify({ dataset_id }),
      },
    ),
  storeInfo: () =>
    request<{ engine: string; path: string; note: string }>('/api/data/store'),
  upload: (file: File) => {
    const fd = new FormData()
    fd.append('file', file)
    return request<{
      row_count: number
      columns: string[]
      preview: Record<string, unknown>[]
      mapping: {
        found: Array<{ key: string; label: string; group: string; required: boolean; source_column: string; mapped: boolean }>
        missing_required: Array<{ key: string; label: string }>
        missing_optional: Array<{ key: string; label: string }>
        unmapped_source_columns: string[]
        coverage: { mapped: number; total_supported: number; required_ok: boolean }
      }
    }>('/api/data/upload', {
      method: 'POST',
      body: fd,
    })
  },
  reportSchema: () =>
    request<{
      description: string
      required_keys: string[]
      fields: Array<{
        key: string
        label: string
        group: string
        required: boolean
        aliases: string[]
      }>
      groups: Record<
        string,
        Array<{ key: string; label: string; required: boolean; aliases: string[] }>
      >
      template_headers: string[]
    }>('/api/data/schema'),
  async downloadTemplate() {
    const token = getToken()
    const res = await fetch('/api/data/template.csv', {
      headers: token ? { Authorization: `Bearer ${token}` } : {},
    })
    if (!res.ok) throw new Error('Could not download template')
    const blob = await res.blob()
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'meta_ads_report_template.csv'
    a.click()
    URL.revokeObjectURL(url)
  },
  metaConnect: (access_token: string, display_name: string) =>
    request<{
      user: { id: string; name: string }
      accounts: Array<{
        id: string
        account_id: string
        name: string
        currency: string
        label: string
      }>
    }>('/api/meta/connect', {
      method: 'POST',
      body: JSON.stringify({ access_token, display_name }),
    }),
  metaFetch: (account_id: string, date_start: string, date_end: string, daily = false) =>
    request<{ row_count: number; account_label: string; preview: Record<string, unknown>[] }>(
      '/api/meta/fetch',
      {
        method: 'POST',
        body: JSON.stringify({ account_id, date_start, date_end, daily }),
      },
    ),
  providers: () =>
    request<{
      options: { id: string; label: string }[]
      defaults: Record<string, string>
      models: Record<string, string[]>
      active: string
      configured: Array<{
        provider: string
        label: string
        model: string
        masked_key: string
        active: boolean
      }>
      usage: {
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
    }>('/api/providers'),
  saveProvider: (body: {
    provider: string
    api_key: string
    model: string
    base_url?: string
    label: string
  }) =>
    request<{ ok: boolean }>('/api/providers', {
      method: 'POST',
      body: JSON.stringify(body),
    }),
  setActiveProvider: (provider: string) =>
    request<{ ok: boolean }>('/api/providers/active', {
      method: 'POST',
      body: JSON.stringify({ provider }),
    }),
  deleteProvider: (provider: string) =>
    request<{ ok: boolean; active: string | null }>(`/api/providers/${encodeURIComponent(provider)}`, {
      method: 'DELETE',
    }),
  resetUsage: () =>
    request<{
      requests: number
      prompt_tokens: number
      completion_tokens: number
      total_tokens: number
      estimated_cost_usd: number
      by_provider: Record<string, unknown>
      recent: unknown[]
    }>('/api/providers/usage', { method: 'DELETE' }),
  dashboard: () =>
    request<{
      kpis: { name: string; value: string }[]
      charts: {
        objective_spend?: { name: string; full_name?: string; value: number; pct: number }[]
        objective_roas?: { name: string; full_name?: string; value: number; spend: number; revenue: number }[]
        objective_spend_revenue?: { name: string; spend: number; revenue: number; roas: number }[]
        campaign_spend?: {
          name: string
          full_name: string
          spend: number
          revenue: number
          roas: number
          ctr: number
        }[]
        ctr_by_campaign?: { name: string; full_name?: string; value: number; spend?: number }[]
        scatter_spend_roas?: { spend: number; roas: number; campaign?: string; purchases?: number }[]
        funnel?: { name: string; value: number; pct: number }[]
        campaign_table?: {
          campaign: string
          spend: number
          revenue: number
          roas: number
          ctr: number
          cpc: number
          clicks: number
          purchases: number
        }[]
        top_ads?: { ad: string; campaign: string; spend: number; roas: number; purchases: number }[]
        weak_ads?: { ad: string; campaign: string; spend: number; roas: number; purchases: number }[]
        status_spend?: { name: string; value: number }[]
        trend?: { date: string; spend: number; revenue: number; roas: number; ctr: number; purchases: number }[]
        adset_perf?: { name: string; full_name: string; spend: number; revenue: number; roas: number; ctr: number }[]
        roas_distribution?: { bucket: string; ads: number; spend: number }[]
        cost_efficiency?: { name: string; full_name: string; cpc: number; cpm: number; spend: number }[]
        reach_frequency?: { name: string; full_name: string; reach: number; frequency: number }[]
      }
      insights: Record<string, string | null>
      row_count: number
      currency: string
      summary?: { campaigns: number; ads: number; objectives: number }
    }>('/api/dashboard'),
  chartInsight: (chartKey: string) =>
    request<{ chart_key: string; insight: string | null; cached: boolean }>(
      `/api/analysis/chart-insight/${chartKey}`,
      { method: 'POST' },
    ),
  detailedAnalysis: () =>
    request<{
      raw: string
      parsed: {
        overall_summary: string
        priority_blocks: { priority: string; title: string; details: string }[]
      } | null
    }>('/api/analysis/detailed', { method: 'POST' }),
  getChat: () => request<{ messages: { role: string; content: string }[] }>('/api/chat'),
  sendChat: (message: string) =>
    request<{ reply: string; messages: { role: string; content: string }[] }>('/api/chat', {
      method: 'POST',
      body: JSON.stringify({ message }),
    }),
  clearChat: () => request<{ ok: boolean }>('/api/chat', { method: 'DELETE' }),
  async downloadPdf() {
    const token = getToken()
    const res = await fetch('/api/report/pdf', {
      method: 'POST',
      headers: token ? { Authorization: `Bearer ${token}` } : {},
    })
    if (!res.ok) {
      let detail = 'PDF generation failed'
      try {
        const body = await res.json()
        if (body?.detail) detail = typeof body.detail === 'string' ? body.detail : detail
      } catch {
        /* ignore */
      }
      throw new Error(detail)
    }
    const blob = await res.blob()
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'Meta_Report.pdf'
    a.click()
    URL.revokeObjectURL(url)
  },
}
