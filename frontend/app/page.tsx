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

type Trace = {
  trace?: Array<{ iteration: number; node: string; action: string }>;
};

export default function Home() {
  const [inv, setInv] = useState<Investigation | null>(null);
  const [trace, setTrace] = useState<Trace | null>(null);
  const [loading, setLoading] = useState(false);
  const [desc, setDesc] = useState("Inspect for surface defects or scratches");
  const [files, setFiles] = useState<File[]>([]);
  const [error, setError] = useState<string | null>(null);

  async function readJson<T>(response: Response): Promise<T> {
    const body = await response.json().catch(() => ({}));
    if (!response.ok) {
      const detail = typeof body.detail === "string" ? body.detail : "Request failed";
      throw new Error(detail);
    }
    return body as T;
  }

  async function createAndRun() {
    setLoading(true);
    setError(null);
    try {
      const created = await fetch(`${API}/investigations`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ textual_description: desc }),
      }).then((r) => readJson<Investigation>(r));

      if (files.length > 0) {
        const form = new FormData();
        files.forEach((file) => form.append("files", file));
        await fetch(`${API}/investigations/${created.investigation_id}/images`, {
          method: "POST",
          body: form,
        }).then((r) => readJson(r));
      }

      const result = await fetch(`${API}/investigations/${created.investigation_id}/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}),
      }).then((r) => readJson<Investigation>(r));
      setInv(result);

      const nextTrace = await fetch(`${API}/investigations/${created.investigation_id}/trace`).then((r) =>
        readJson<Trace>(r),
      );
      setTrace(nextTrace);
    } catch (e) {
      console.error(e);
      setError(e instanceof Error ? e.message : "API error - is the backend running on :8000?");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="min-h-screen bg-slate-950 p-6 font-sans text-slate-100 md:p-8">
      <div className="mx-auto max-w-6xl">
        <header className="mb-8">
          <h1 className="text-3xl font-bold tracking-normal">Multimodal AI Investigation Agent</h1>
          <p className="mt-2 text-slate-400">Planner / Executor / Verifier / Uncertainty-aware</p>
        </header>

        <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
          <section className="rounded-lg border border-slate-800 bg-slate-900 p-6">
            <h2 className="mb-4 text-lg font-semibold">New Investigation</h2>
            <textarea
              className="mb-4 min-h-24 w-full rounded-md border border-slate-700 bg-slate-800 p-3 text-sm text-slate-100 outline-none focus:border-sky-500"
              value={desc}
              onChange={(e) => setDesc(e.target.value)}
            />
            <label className="mb-4 block">
              <span className="mb-2 block text-sm font-medium text-slate-300">Images</span>
              <input
                className="block w-full cursor-pointer rounded-md border border-slate-700 bg-slate-800 text-sm text-slate-300 file:mr-4 file:border-0 file:bg-sky-600 file:px-3 file:py-2 file:text-sm file:font-medium file:text-white hover:file:bg-sky-500"
                type="file"
                accept="image/png,image/jpeg,image/webp"
                multiple
                onChange={(e) => setFiles(Array.from(e.currentTarget.files ?? []))}
              />
            </label>
            {files.length > 0 && (
              <div className="mb-4 rounded-md border border-slate-800 bg-slate-950 p-3 text-xs text-slate-400">
                {files.map((file) => (
                  <div key={`${file.name}-${file.size}`}>{file.name}</div>
                ))}
              </div>
            )}
            <button
              onClick={createAndRun}
              disabled={loading}
              className="w-full rounded-md bg-sky-600 py-2 font-medium text-white hover:bg-sky-500 disabled:cursor-not-allowed disabled:opacity-50"
            >
              {loading ? "Investigating..." : "Run Investigation"}
            </button>
            {error && <p className="mt-3 rounded-md bg-red-950 p-3 text-xs leading-5 text-red-200">{error}</p>}
          </section>

          <section className="rounded-lg border border-slate-800 bg-slate-900 p-6 lg:col-span-2">
            <h2 className="mb-4 text-lg font-semibold">Decision & Confidence</h2>
            {inv ? (
              <div className="space-y-3">
                <div className="flex flex-wrap gap-3 text-sm">
                  <span className="rounded-full bg-slate-800 px-3 py-1">Status: {inv.status}</span>
                  <span className="rounded-full bg-sky-950 px-3 py-1">Decision: {inv.final_decision ?? "-"}</span>
                  <span className="rounded-full bg-slate-800 px-3 py-1">
                    Confidence: {inv.confidence?.overall?.toFixed(2) ?? "-"} ({inv.confidence?.level ?? "-"})
                  </span>
                  <span className="rounded-full bg-slate-800 px-3 py-1">Iters: {inv.iteration}</span>
                </div>
                {inv.explanation && (
                  <pre className="max-h-72 overflow-auto whitespace-pre-wrap rounded-md bg-slate-950 p-4 text-xs leading-5">
                    {inv.explanation}
                  </pre>
                )}
              </div>
            ) : (
              <p className="text-slate-500">No investigation yet.</p>
            )}
          </section>

          <section className="rounded-lg border border-slate-800 bg-slate-900 p-6 lg:col-span-3">
            <h2 className="mb-4 text-lg font-semibold">Investigation Trace</h2>
            {trace ? (
              <ol className="space-y-1 font-mono text-sm">
                {trace.trace?.map((t, i) => (
                  <li key={`${t.iteration}-${t.node}-${t.action}-${i}`} className="text-slate-300">
                    <span className="text-slate-500">[{t.iteration}]</span> {t.node} -&gt; {t.action}
                  </li>
                ))}
              </ol>
            ) : (
              <p className="text-slate-500">Trace appears after a run.</p>
            )}
          </section>
        </div>
      </div>
    </main>
  );
}
