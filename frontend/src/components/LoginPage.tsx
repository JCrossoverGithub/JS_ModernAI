import { useState } from "react";
import { useAuth } from "../context/AuthContext";
import { login, register } from "../services/api";
import { Sparkles, ArrowRight, Eye, EyeOff } from "lucide-react";

export default function LoginPage() {
  const { setUser } = useAuth();
  const [isRegistering, setIsRegistering] = useState(false);
  const [username, setUsername] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setLoading(true);
    try {
      const res = isRegistering
        ? await register(username, email, password)
        : await login(email, password);
      setUser(res);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex bg-[#0a0b12]">
      {/* Left panel — branding */}
      <div className="hidden lg:flex lg:w-1/2 items-center justify-center p-12 relative overflow-hidden">
        {/* Gradient orbs */}
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-indigo-500/10 rounded-full blur-[120px]" />
        <div className="absolute bottom-1/4 right-1/4 w-80 h-80 bg-violet-500/10 rounded-full blur-[100px]" />

        <div className="relative z-10 max-w-md">
          <div className="inline-flex items-center justify-center w-14 h-14 rounded-2xl bg-gradient-to-br from-indigo-500/20 to-violet-500/20 border border-indigo-500/20 mb-8">
            <Sparkles className="text-indigo-400" size={24} />
          </div>
          <h1 className="text-4xl font-bold text-white mb-4 tracking-tight">
            LibraryAI Research
          </h1>
          <p className="text-lg text-slate-400 leading-relaxed mb-8">
            Your AI-powered research assistant. Upload documents, search the
            web, and build your personal knowledge base with intelligent
            conversations.
          </p>
          <div className="space-y-4">
            {[
              "Upload and search your document library",
              "AI-powered web research with source citations",
              "Personal memory that learns your context",
              "Export and organize your research threads",
            ].map((feature, i) => (
              <div key={i} className="flex items-center gap-3">
                <div className="w-1.5 h-1.5 rounded-full bg-indigo-400" />
                <span className="text-sm text-slate-300">{feature}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Right panel — form */}
      <div className="flex-1 flex items-center justify-center p-8">
        <div className="w-full max-w-sm">
          {/* Mobile logo */}
          <div className="lg:hidden text-center mb-8">
            <div className="inline-flex items-center justify-center w-12 h-12 rounded-xl bg-gradient-to-br from-indigo-500/20 to-violet-500/20 border border-indigo-500/20 mb-4">
              <Sparkles className="text-indigo-400" size={20} />
            </div>
            <h1 className="text-2xl font-bold text-white">LibraryAI</h1>
          </div>

          <h2 className="text-xl font-semibold text-white mb-1">
            {isRegistering ? "Create account" : "Welcome back"}
          </h2>
          <p className="text-sm text-slate-400 mb-6">
            {isRegistering
              ? "Set up your research workspace"
              : "Sign in to your research workspace"}
          </p>

          {error && (
            <div className="flex items-start gap-2 bg-red-500/10 border border-red-500/20 text-red-300 px-4 py-3 rounded-xl mb-5 text-sm">
              <span className="shrink-0 mt-0.5">!</span>
              <span>{error}</span>
            </div>
          )}

          <form onSubmit={handleSubmit} className="space-y-3">
            {isRegistering && (
              <div>
                <label className="block text-xs font-medium text-slate-400 mb-1.5">
                  Username
                </label>
                <input
                  type="text"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  required
                  className="w-full px-4 py-3 bg-white/[0.04] border border-white/[0.08] rounded-xl text-sm text-white placeholder-slate-500 outline-none focus:border-indigo-500/50 transition-colors"
                  placeholder="Your name"
                />
              </div>
            )}
            <div>
              <label className="block text-xs font-medium text-slate-400 mb-1.5">
                Email
              </label>
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
                className="w-full px-4 py-3 bg-white/[0.04] border border-white/[0.08] rounded-xl text-sm text-white placeholder-slate-500 outline-none focus:border-indigo-500/50 transition-colors"
                placeholder="you@example.com"
              />
            </div>
            <div>
              <label className="block text-xs font-medium text-slate-400 mb-1.5">
                Password
              </label>
              <div className="relative">
                <input
                  type={showPassword ? "text" : "password"}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  required
                  minLength={6}
                  className="w-full px-4 py-3 pr-10 bg-white/[0.04] border border-white/[0.08] rounded-xl text-sm text-white placeholder-slate-500 outline-none focus:border-indigo-500/50 transition-colors"
                  placeholder="••••••••"
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-500 hover:text-slate-300 transition-colors"
                >
                  {showPassword ? <EyeOff size={16} /> : <Eye size={16} />}
                </button>
              </div>
            </div>

            <button
              type="submit"
              disabled={loading}
              className="w-full flex items-center justify-center gap-2 py-3 mt-2 bg-indigo-500 hover:bg-indigo-600 disabled:bg-indigo-500/50 disabled:cursor-not-allowed text-white rounded-xl font-medium transition-all active:scale-[0.98]"
            >
              {loading ? (
                <span className="inline-block w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
              ) : (
                <>
                  {isRegistering ? "Create Account" : "Sign In"}
                  <ArrowRight size={15} />
                </>
              )}
            </button>
          </form>

          <p className="text-center text-slate-500 text-sm mt-6">
            {isRegistering ? "Already have an account?" : "Don't have an account?"}{" "}
            <button
              onClick={() => {
                setIsRegistering(!isRegistering);
                setError("");
              }}
              className="text-indigo-400 hover:text-indigo-300 font-medium transition-colors"
            >
              {isRegistering ? "Sign in" : "Register"}
            </button>
          </p>
        </div>
      </div>
    </div>
  );
}
