import { NavLink, Outlet } from 'react-router-dom'
import { useAuth } from '../auth'

const links = [
  { to: '/', label: 'Home', end: true },
  { to: '/meta', label: 'Connect Meta API' },
  { to: '/upload', label: 'Upload Report' },
  { to: '/dashboard', label: 'Dashboard + Chat' },
  { to: '/providers', label: 'AI Providers' },
]

export function Shell() {
  const { session, logout } = useAuth()

  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="brand">
          <strong>Meta Analyst</strong>
          <span>Ads performance</span>
        </div>

        <nav className="nav">
          {links.map((l) => (
            <NavLink key={l.to} to={l.to} end={l.end} className={({ isActive }) => (isActive ? 'active' : '')}>
              {l.label}
            </NavLink>
          ))}
        </nav>

        <div className="sidebar-meta">
          <div className="meta-block">
            {session?.meta.account_label ? (
              <>
                <strong>Meta</strong>
                <div>{session.meta.account_label}</div>
              </>
            ) : (
              <div>No Meta account selected</div>
            )}
          </div>
          <div className="meta-block">
            {session?.provider ? (
              <>
                <strong>AI</strong>
                <div>
                  {session.provider.label} · {session.provider.model}
                </div>
              </>
            ) : (
              <div>No AI provider active</div>
            )}
          </div>
          {session?.has_data && (
            <div className="meta-block">
              <strong>Data</strong>
              <div>{session.row_count.toLocaleString()} rows loaded</div>
            </div>
          )}
          <button className="btn btn-sidebar" type="button" onClick={() => logout()}>
            Logout
          </button>
        </div>
      </aside>
      <main className="main">
        <Outlet />
      </main>
    </div>
  )
}
