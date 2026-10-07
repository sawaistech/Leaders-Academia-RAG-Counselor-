// Leaders Academia assistant on Cloudflare Workers: web chat + WhatsApp Cloud API webhook.
import DATA from "../data.json";
import PAGE from "./page.html";
import SYSTEM from "./system_prompt.txt";

const CHUNKS = DATA.chunks;
const CATALOG = DATA.catalog;
const EMB_MODEL = DATA.emb_model;
const HAS_DENSE = !!(EMB_MODEL && CHUNKS.length && CHUNKS[0].vec && CHUNKS[0].vec.length);
const FALLBACK_MSG = "Sorry, I'm having a little trouble right now. Please try again in a moment! 🙏";

const json = (obj, status = 200) =>
  new Response(JSON.stringify(obj), { status, headers: { "content-type": "application/json" } });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// ---------------------------------------------------------------- BM25 (keyword search)
const tok = (s) => s.toLowerCase().match(/[\p{L}\p{N}_]+/gu) || [];
const docs = CHUNKS.map((c) => tok(`${c.type}: ${c.title}\n${c.text}`));
const N = docs.length;
const avgdl = docs.reduce((a, d) => a + d.length, 0) / N;
const df = new Map();
const tfs = docs.map((d) => {
  const m = new Map();
  for (const w of d) m.set(w, (m.get(w) || 0) + 1);
  for (const w of m.keys()) df.set(w, (df.get(w) || 0) + 1);
  return m;
});
function bm25Scores(query) {
  const q = tok(query), k1 = 1.5, b = 0.75;
  return docs.map((d, i) => {
    let s = 0;
    for (const w of q) {
      const f = tfs[i].get(w);
      if (!f) continue;
      const n = df.get(w);
      const idf = Math.log((N - n + 0.5) / (n + 0.5) + 1);
      s += (idf * f * (k1 + 1)) / (f + k1 * (1 - b + (b * d.length) / avgdl));
    }
    return s;
  });
}
const topIdx = (scores, pool = 20, positiveOnly = false) =>
  scores.map((s, i) => [s, i]).filter((x) => !positiveOnly || x[0] > 0)
    .sort((a, b) => b[0] - a[0]).slice(0, pool).map((x) => x[1]);

// ---------------------------------------------------------------- Gemini / Groq
async function embedQuery(text, env) {
  const r = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/${EMB_MODEL}:embedContent`, {
    method: "POST",
    headers: { "content-type": "application/json", "x-goog-api-key": env.GEMINI_API_KEY },
    body: JSON.stringify({
      model: `models/${EMB_MODEL}`, content: { parts: [{ text }] },
      taskType: "RETRIEVAL_QUERY", outputDimensionality: CHUNKS[0].vec.length,
    }),
  });
  if (!r.ok) throw new Error(`embed ${r.status}`);
  const v = (await r.json()).embedding.values;
  const norm = Math.sqrt(v.reduce((a, x) => a + x * x, 0)) + 1e-9;
  return v.map((x) => x / norm);
}

async function gemini(prompt, system, temperature, env) {
  if (!env.GEMINI_API_KEY) throw new Error("GEMINI_API_KEY not set");
  const models = (env.GEMINI_MODELS || "gemini-2.5-flash,gemini-2.5-flash-lite").split(",").map((s) => s.trim()).filter(Boolean);
  let last;
  for (const model of models) {
    for (let attempt = 0; attempt < 2; attempt++) {
      const body = { contents: [{ role: "user", parts: [{ text: prompt }] }], generationConfig: { temperature } };
      if (system) body.systemInstruction = { parts: [{ text: system }] };
      const r = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent`, {
        method: "POST",
        headers: { "content-type": "application/json", "x-goog-api-key": env.GEMINI_API_KEY },
        body: JSON.stringify(body),
      });
      if (r.ok) {
        const j = await r.json();
        const text = (j.candidates?.[0]?.content?.parts || []).map((p) => p.text || "").join("").trim();
        if (text) return text;
        last = new Error("empty response"); break;
      }
      last = new Error(`Gemini ${model} HTTP ${r.status}`);
      if ([429, 500, 502, 503, 504].includes(r.status)) { await sleep(700 * (attempt + 1)); continue; }
      break;                                   // 404/400 etc: try the next model
    }
  }
  throw last || new Error("Gemini unavailable");
}

async function groq(prompt, system, temperature, env) {
  if (!env.GROQ_API_KEY) throw new Error("GROQ_API_KEY not set");
  const messages = [...(system ? [{ role: "system", content: system }] : []), { role: "user", content: prompt }];
  const r = await fetch("https://api.groq.com/openai/v1/chat/completions", {
    method: "POST",
    headers: { "content-type": "application/json", authorization: `Bearer ${env.GROQ_API_KEY}` },
    body: JSON.stringify({ model: env.GROQ_MODEL || "llama-3.3-70b-versatile", messages, temperature }),
  });
  if (!r.ok) throw new Error(`Groq HTTP ${r.status}`);
  return ((await r.json()).choices?.[0]?.message?.content || "").trim();
}

async function llm(prompt, system, temperature, env) {
  try { return await gemini(prompt, system, temperature, env); }
  catch (e) { console.log("Gemini failed:", String(e)); return await groq(prompt, system, temperature, env); }
}

// ---------------------------------------------------------------- Retrieval + RAG
async function retrieve(query, env, k = 6) {
  const rankings = [topIdx(bm25Scores(query), 20, true)];
  if (HAS_DENSE && env.GEMINI_API_KEY) {
    try {
      const qv = await embedQuery(query, env);
      const sims = CHUNKS.map((c) => { let s = 0; for (let i = 0; i < qv.length; i++) s += c.vec[i] * qv[i]; return s; });
      rankings.push(topIdx(sims, 20));
    } catch (e) { console.log("dense retrieval failed:", String(e)); }
  }
  const score = new Map();                      // Reciprocal Rank Fusion
  for (const ranking of rankings)
    ranking.forEach((idx, r) => score.set(idx, (score.get(idx) || 0) + 1 / (60 + r)));
  return [...score.entries()].sort((a, b) => b[1] - a[1]).slice(0, k).map(([i]) => CHUNKS[i]);
}

async function ask(question, history, env) {
  let standalone = question;
  if (history && history.length) {
    const convo = history.slice(-6).map((h) => `${h.role}: ${h.content}`).join("\n");
    try {
      standalone = (await llm(
        "Rewrite the final user question as a standalone question, keeping its language. Return only the question.\n\n" +
        `${convo}\nuser: ${question}`, null, 0, env)) || question;
    } catch { standalone = question; }
  }
  const docsHit = await retrieve(standalone, env, 6);
  const details = docsHit.map((d) => `[${d.type}: ${d.title}]\n${d.text}`).join("\n\n---\n\n");
  const prompt = `FULL PROGRAM CATALOG:\n${CATALOG}\n\nRELEVANT DETAILS:\n${details}\n\nSTUDENT'S MESSAGE: ${question}`;
  return await llm(prompt, SYSTEM, 0.5, env);
}

// ---------------------------------------------------------------- Rate limit + memory
const buckets = new Map();
function allow(key, limit, windowMs = 3600000) {
  const now = Date.now();
  const q = (buckets.get(key) || []).filter((t) => now - t < windowMs);
  if (q.length >= limit) { buckets.set(key, q); return false; }
  q.push(now); buckets.set(key, q);
  if (buckets.size > 5000) buckets.delete(buckets.keys().next().value);
  return true;
}
const memHist = new Map();
async function getHist(env, id) {
  if (env.CHAT_KV) { try { return (await env.CHAT_KV.get("h:" + id, "json")) || []; } catch { return []; } }
  return memHist.get(id) || [];
}
async function putHist(env, id, h) {
  h = h.slice(-8);
  if (env.CHAT_KV) { try { await env.CHAT_KV.put("h:" + id, JSON.stringify(h), { expirationTtl: 21600 }); } catch {} return; }
  memHist.set(id, h);
  if (memHist.size > 500) memHist.delete(memHist.keys().next().value);
}

// ---------------------------------------------------------------- Web chat API
async function handleWebChat(request, env) {
  const ip = request.headers.get("cf-connecting-ip") || "x";
  if (!allow("web:" + ip, parseInt(env.WEB_MAX_PER_HOUR || "40")))
    return json({ reply: "You've sent a lot of messages! Please try again a little later, or email info@leadersacademia.com." });
  let body;
  try { body = await request.json(); } catch { return json({ reply: "Please type your question." }, 400); }
  const msg = String(body.message || "").slice(0, 1000).trim();
  if (!msg) return json({ reply: "Please type your question and I'll be happy to help!" });
  const hist = (Array.isArray(body.history) ? body.history : []).slice(-6).filter((h) => h && typeof h === "object")
    .map((h) => ({ role: h.role === "assistant" ? "assistant" : "user", content: String(h.content || "").slice(0, 1000) }));
  try { return json({ reply: await ask(msg, hist, env) }); }
  catch (e) { console.log("web chat failed:", String(e)); return json({ reply: FALLBACK_MSG }); }
}

// ---------------------------------------------------------------- WhatsApp Cloud API
const waFormat = (t) => t.replace(/\*\*(.+?)\*\*/g, "*$1*").replace(/^#{1,6}\s*/gm, "").trim();
function waSplit(text, limit = 3500) {
  const out = []; let cur = "";
  for (const para of text.split("\n\n")) {
    if (cur && cur.length + para.length + 2 > limit) { out.push(cur); cur = ""; }
    cur += (cur ? "\n\n" : "") + para;
  }
  if (cur) out.push(cur);
  return out.length ? out : [text];
}
async function waPost(env, payload) {
  const r = await fetch(`https://graph.facebook.com/${env.GRAPH_VERSION || "v21.0"}/${env.PHONE_NUMBER_ID}/messages`, {
    method: "POST",
    headers: { authorization: `Bearer ${env.WHATSAPP_TOKEN}`, "content-type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!r.ok) console.log("WhatsApp send failed", r.status, await r.text());
}
async function waSend(env, to, text) {
  for (const part of waSplit(waFormat(text)))
    await waPost(env, { messaging_product: "whatsapp", to, type: "text", text: { body: part } });
}

const seen = new Map();
async function handleWhatsApp(payload, env) {
  for (const entry of payload.entry || []) for (const change of entry.changes || [])
    for (const msg of change.value?.messages || []) {
      const { id: mid, from: sender } = msg;
      if (!mid || !sender || seen.has(mid)) continue;
      seen.set(mid, 1); if (seen.size > 2000) seen.delete(seen.keys().next().value);
      try { await waPost(env, { messaging_product: "whatsapp", status: "read", message_id: mid }); } catch {}
      try {
        if (msg.type !== "text") {
          await waSend(env, sender, "Thanks for your message! 😊 I can only read text messages right now, so please type your question and I'll gladly help.");
          continue;
        }
        if (!allow("wa:" + sender, parseInt(env.MAX_MSGS_PER_HOUR || "30"))) {
          await waSend(env, sender, "You've sent quite a few messages! Please try again a little later, or email us at info@leadersacademia.com.");
          continue;
        }
        const text = msg.text.body;
        const hist = await getHist(env, sender);
        const answer = await ask(text, hist, env);
        await putHist(env, sender, [...hist, { role: "user", content: text }, { role: "assistant", content: answer }]);
        await waSend(env, sender, answer);
      } catch (e) {
        console.log("WhatsApp handling failed:", String(e));
        try { await waSend(env, sender, FALLBACK_MSG); } catch {}
      }
    }
}

async function validSignature(buf, header, secret) {
  const key = await crypto.subtle.importKey("raw", new TextEncoder().encode(secret),
    { name: "HMAC", hash: "SHA-256" }, false, ["sign"]);
  const sig = new Uint8Array(await crypto.subtle.sign("HMAC", key, buf));
  const expected = "sha256=" + [...sig].map((b) => b.toString(16).padStart(2, "0")).join("");
  const got = header || "";
  if (got.length !== expected.length) return false;
  let diff = 0;
  for (let i = 0; i < expected.length; i++) diff |= expected.charCodeAt(i) ^ got.charCodeAt(i);
  return diff === 0;
}

// ---------------------------------------------------------------- Router
export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);
    const { pathname } = url;
    try {
      if (request.method === "GET" && pathname === "/")
        return new Response(PAGE, { headers: { "content-type": "text/html; charset=utf-8" } });
      if (request.method === "GET" && pathname === "/health")
        return json({ status: "ok", chunks: N, dense_retrieval: HAS_DENSE && !!env.GEMINI_API_KEY, kv_memory: !!env.CHAT_KV });
      if (request.method === "POST" && pathname === "/api/chat") return await handleWebChat(request, env);

      if (pathname === "/webhook" && request.method === "GET") {
        const p = url.searchParams;
        if (p.get("hub.mode") === "subscribe" && env.VERIFY_TOKEN && p.get("hub.verify_token") === env.VERIFY_TOKEN)
          return new Response(p.get("hub.challenge") || "");
        return new Response("Forbidden", { status: 403 });
      }
      if (pathname === "/webhook" && request.method === "POST") {
        const buf = await request.arrayBuffer();
        if (env.APP_SECRET && !(await validSignature(buf, request.headers.get("x-hub-signature-256"), env.APP_SECRET)))
          return json({ error: "bad signature" }, 403);
        let payload;
        try { payload = JSON.parse(new TextDecoder().decode(buf)); } catch { return json({ error: "bad json" }, 400); }
        ctx.waitUntil(handleWhatsApp(payload, env));   // reply after answering Meta with 200
        return json({ status: "received" });
      }
      return new Response("Not found", { status: 404 });
    } catch (e) {
      console.log("unhandled:", String(e));
      return json({ error: "server error" }, 500);
    }
  },
};
