import { createContext, useCallback, useContext, useEffect, useState } from "react";
import { getMe, getToken, login as apiLogin, logout as apiLogout, register as apiRegister } from "../lib/api";

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  const refreshMe = useCallback(async () => {
    if (!getToken()) {
      setUser(null);
      setLoading(false);
      return;
    }
    try {
      setUser(await getMe());
    } catch {
      setUser(null);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refreshMe();
  }, [refreshMe]);

  // Any API call that gets a 401 (missing/expired token) dispatches this —
  // force the user back to the login screen instead of a broken app state.
  useEffect(() => {
    const onUnauthorized = () => setUser(null);
    window.addEventListener("auth:unauthorized", onUnauthorized);
    return () => window.removeEventListener("auth:unauthorized", onUnauthorized);
  }, []);

  const login = useCallback(async (username, password) => {
    await apiLogin(username, password);
    await refreshMe();
  }, [refreshMe]);

  const register = useCallback(async (username, password, email) => {
    await apiRegister(username, password, email);
    await apiLogin(username, password);
    await refreshMe();
  }, [refreshMe]);

  const logout = useCallback(() => {
    apiLogout();
    setUser(null);
  }, []);

  return (
    <AuthContext.Provider value={{ user, loading, isAuthenticated: !!user, login, register, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within an AuthProvider");
  return ctx;
}
