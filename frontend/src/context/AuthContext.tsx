/**
 * AuthContext.tsx — Global authentication state management.
 *
 * Provides user info (token, username, userId) to all components via React Context.
 * Persists auth state to localStorage so the user stays logged in across page reloads.
 * Components use the useAuth() hook to access the current user or trigger logout.
 */

import {
  createContext,
  useContext,
  useState,
  useEffect,
  type ReactNode,
} from "react";
import type { AuthResponse } from "../types";

interface AuthContextType {
  user: AuthResponse | null;
  setUser: (user: AuthResponse | null) => void;
  logout: () => void;
}

const AuthContext = createContext<AuthContextType>({
  user: null,
  setUser: () => {},
  logout: () => {},
});

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUserState] = useState<AuthResponse | null>(null);

  useEffect(() => {
    const stored = localStorage.getItem("user");
    if (stored) {
      try {
        setUserState(JSON.parse(stored));
      } catch {
        localStorage.removeItem("user");
        localStorage.removeItem("token");
      }
    }
  }, []);

  const setUser = (u: AuthResponse | null) => {
    setUserState(u);
    if (u) {
      localStorage.setItem("user", JSON.stringify(u));
      localStorage.setItem("token", u.token);
    } else {
      localStorage.removeItem("user");
      localStorage.removeItem("token");
    }
  };

  const logout = () => setUser(null);

  return (
    <AuthContext.Provider value={{ user, setUser, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

export const useAuth = () => useContext(AuthContext);
