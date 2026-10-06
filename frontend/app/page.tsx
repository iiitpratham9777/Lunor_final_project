"use client";

import { useState } from "react";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api/v1";

type Investigation = {
  investigation_id: string;
  status: string;
  final_decision?: string;
  confidence?: { overall: number; level: string };
  explanation?: string;
  iteration: number;
};

export default function Home() {
  const [inv, setInv] = useState<Investigation | null>(null);
  const [trace, setTrace] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [desc, setDesc] = useState("Inspect for surface defects or scratches");

  async function createAndRun() {
    setLoading(true);
    try {
      const c = await fetch(`${API}/investigations`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ textual_description: desc }),
      }).then((r) => r.json());

      // Note: image upload via form in full UI; text-only path for scaffold
      const run = await fetch(`${API}/investigations/${c.investigation_id}/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}),
      }).then((r) => r.json());
      setInv(run);

      const t = await fetch(`${API}/investigations/${c.investigation_id}/trace`).then((r) => r.json());
      setTrace(t);
    } catch (e) {
      console.error(e);
      alert("API error — is the backend running on :8000?");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="min-h-screen bg-slate-950 text-slate-100 p-8 font-sans">
      <h1 className="text-3xl font-bold mb-2">Multimodal AI Investigation Agent</h1>
      <p className="text-slate-400 mb-8">Planner · Executor · Verifier · Uncertainty-aware</p>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <section className="bg-slate-900 rounded-xl p-6 border border-slate-800">
          <h2 className="text-lg font-semibold mb-4">New Investigation</h2>
          <textarea
            className="w-full bg-slate-800 border border-slate-700 rounded p-3 text-sm mb-4"
            rows={3}
            value={desc}
            onChange={(e) => setDesc(e.target.value)}
          />
          <button
            onClick={createAndRun}
            disabled={loading}
            className="w-full bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 rounded-lg py-2 font-medium"
          >
            {loading ? "Investigating…" : "Run Investigation"}
          </button>
          <p className="text-xs text-slate-500 mt-3">
            Upload images via API <code>POST /investigations/&#123;id&#125;/images</code> for full vision path.
          </p>
        </section>

        <section className="bg-slate-900 rounded-xl p-6 border border-slate-800 lg:col-span-2">
          <h2 className="text-lg font-semibold mb-4">Decision & Confidence</h2>
          {inv ? (
            <div className="space-y-3">
              <div className="flex gap-4 text-sm">
                <span className="px-3 py-1 rounded-full bg-slate-800">Status: {inv.status}</span>
                <span className="px-3 py-1 rounded-full bg-indigo-900/50">
                  Decision: {inv.final_decision ?? "—"}
                </span>
                <span className="px-3 py-1 rounded-full bg-slate-800">
                  Confidence: {inv.confidence?.overall?.toFixed(2) ?? "—"} ({inv.confidence?.level})
                </span>
                <span className="px-3 py-1 rounded-full bg-slate-800">Iters: {inv.iteration}</span>
              </div>
              {inv.explanation && (
                <pre className="text-xs bg-slate-950 p-4 rounded-lg overflow-auto max-h-64 whitespace-pre-wrap">
                  {inv.explanation}
                </pre>
              )}
            </div>
          ) : (
            <p className="text-slate-500">No investigation yet.</p>
          )}
        </section>

        <section className="bg-slate-900 rounded-xl p-6 border border-slate-800 lg:col-span-3">
          <h2 className="text-lg font-semibold mb-4">Investigation Trace</h2>
          {trace ? (
            <ol className="text-sm space-y-1 font-mono">
              {trace.trace?.map((t: any, i: number) => (
                <li key={i} className="text-slate-300">
                  <span className="text-slate-500">[{t.iteration}]</span> {t.node} → {t.action}
                </li>
              ))}
            </ol>
          ) : (
            <p className="text-slate-500">Trace appears after a run.</p>
          )}
        </section>
      </div>
    </main>
  );
}
