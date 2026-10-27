import { Navigate, Route, Routes } from 'react-router-dom'
import type { ReactNode } from 'react'
import { AuthProvider, useAuth } from './auth'
import { Shell } from './layout/Shell'
import { Dashboard } from './pages/Dashboard'
import { Home } from './pages/Home'
import { Login } from './pages/Login'
import { MetaConnect } from './pages/MetaConnect'
import { Providers } from './pages/Providers'
import { Upload } from './pages/Upload'

function Protected({ children }: { children: ReactNode }) {
  const { ready, token } = useAuth()
  if (!ready) return <div className="login-page">Loading…</div>
  if (!token) return <Navigate to="/login" replace />
  return children
}

export default function App() {
  return (
    <AuthProvider>
      <Routes>
        <Route path="/login" element={<Login />} />
        <Route
          path="/"
          element={
            <Protected>
              <Shell />
            </Protected>
          }
        >
          <Route index element={<Home />} />
          <Route path="meta" element={<MetaConnect />} />
          <Route path="upload" element={<Upload />} />
          <Route path="dashboard" element={<Dashboard />} />
          <Route path="chat" element={<Navigate to="/dashboard" replace />} />
          <Route path="providers" element={<Providers />} />
        </Route>
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </AuthProvider>
  )
}
