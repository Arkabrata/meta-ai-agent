import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from 'react'
import { api, getToken, setToken } from './api'

type SessionInfo = Awaited<ReturnType<typeof api.session>>

type AuthContextValue = {
  ready: boolean
  token: string | null
  session: SessionInfo | null
  login: (password: string) => Promise<void>
  logout: () => Promise<void>
  refreshSession: () => Promise<void>
}

const AuthContext = createContext<AuthContextValue | null>(null)

export function AuthProvider({ children }: { children: ReactNode }) {
  const [ready, setReady] = useState(false)
  const [token, setTokenState] = useState<string | null>(getToken())
  const [session, setSession] = useState<SessionInfo | null>(null)

  const refreshSession = useCallback(async () => {
    if (!getToken()) {
      setSession(null)
      return
    }
    try {
      const info = await api.session()
      setSession(info)
    } catch {
      setToken(null)
      setTokenState(null)
      setSession(null)
    }
  }, [])

  useEffect(() => {
    ;(async () => {
      if (getToken()) {
        await refreshSession()
      }
      setReady(true)
    })()
  }, [refreshSession])

  const login = useCallback(async (password: string) => {
    const res = await api.login(password)
    setToken(res.token)
    setTokenState(res.token)
    await refreshSession()
  }, [refreshSession])

  const logout = useCallback(async () => {
    try {
      await api.logout()
    } catch {
      /* ignore */
    }
    setToken(null)
    setTokenState(null)
    setSession(null)
  }, [])

  const value = useMemo(
    () => ({ ready, token, session, login, logout, refreshSession }),
    [ready, token, session, login, logout, refreshSession],
  )

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}

export function useAuth() {
  const ctx = useContext(AuthContext)
  if (!ctx) throw new Error('useAuth requires AuthProvider')
  return ctx
}
