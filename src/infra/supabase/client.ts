import { createBrowserClient } from "@supabase/ssr";

/**
 * Browser Supabase client. Returns `null` when Supabase is not configured so
 * that auth is genuinely optional (the app must run without credentials).
 * Callers MUST handle the null case (treat as logged-out / auth-disabled).
 */
export function createClient() {
  const url = process.env.NEXT_PUBLIC_SUPABASE_URL;
  const key = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;
  if (!url || !key) return null;
  return createBrowserClient(url, key);
}
