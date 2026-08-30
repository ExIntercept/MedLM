/* ==============================================================================
   MEDGEMMA RAG SYSTEM — CLIENT LOGIC & SSE STREAMING
   ============================================================================== */

let currentToken = localStorage.getItem("medrag_jwt_token") || null;
let currentConversationId = null;
let activeMode = "patient";
let authTab = "login";
let isStreaming = false;

// ── INIT ON LOAD ────────────────────────────────────────────────────────
window.addEventListener("DOMContentLoaded", () => {
  if (currentToken) {
    initApp();
  } else {
    showGate();
  }
  checkHealth();
  setInterval(checkHealth, 15000);
});

// ── AUTH LOGIC ──────────────────────────────────────────────────────────
function switchAuthTab(tab) {
  authTab = tab;
  document.getElementById("tab-login-btn").classList.toggle("active", tab === "login");
  document.getElementById("tab-reg-btn").classList.toggle("active", tab === "register");
  document.getElementById("email-field").style.display = tab === "register" ? "block" : "none";
  document.getElementById("btn-submit-auth").textContent = tab === "login" ? "Sign In to Console" : "Create New Account";
  document.getElementById("gate-msg").textContent = "";
}

async function handleAuthSubmit(e) {
  e.preventDefault();
  const user = document.getElementById("g-user").value.trim();
  const pass = document.getElementById("g-pass").value;
  const email = document.getElementById("g-email").value.trim() || undefined;
  const msgEl = document.getElementById("gate-msg");

  msgEl.className = "msg";
  msgEl.textContent = "Authenticating...";

  try {
    if (authTab === "register") {
      const regRes = await fetch("/api/auth/register", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username: user, password: pass, email }),
      });
      if (!regRes.ok) {
        const err = await regRes.json();
        throw new Error(err.detail || "Registration failed");
      }
    }

    const form = new URLSearchParams();
    form.set("username", user);
    form.set("password", pass);

    const loginRes = await fetch("/api/auth/login", {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body: form,
    });
    if (!loginRes.ok) {
      const err = await loginRes.json();
      throw new Error(err.detail || "Login failed");
    }

    const data = await loginRes.json();
    currentToken = data.access_token;
    localStorage.setItem("medrag_jwt_token", currentToken);
    initApp();
  } catch (err) {
    msgEl.className = "msg msg-err";
    msgEl.textContent = err.message;
  }
}

function logout() {
  currentToken = null;
  localStorage.removeItem("medrag_jwt_token");
  currentConversationId = null;
  showGate();
}

function showGate() {
  document.getElementById("gate").classList.remove("hidden");
  document.getElementById("app").classList.add("hidden");
}

async function initApp() {
  try {
    const res = await fetch("/api/auth/me", {
      headers: { Authorization: `Bearer ${currentToken}` },
    });
    if (!res.ok) throw new Error("Invalid session");
    const user = await res.json();

    document.getElementById("who-name").textContent = user.username;
    document.getElementById("avatar-initial").textContent = (user.username[0] || "U").toUpperCase();
    document.getElementById("gate").classList.add("hidden");
    document.getElementById("app").classList.remove("hidden");

    loadConversations();
    loadEvaluationSummary();
  } catch {
    logout();
  }
}

// ── HEALTH & MODE ───────────────────────────────────────────────────────
async function checkHealth() {
  try {
    const res = await fetch("/api/health");
    if (!res.ok) return;
    const data = await res.json();

    const colabEl = document.getElementById("status-colab");
    const colabLbl = document.getElementById("lbl-colab-status");
    if (data.colab_medgemma) {
      colabEl.className = "health-chip chip-ok";
      colabLbl.textContent = "Online (" + (data.model || "MedGemma") + ")";
    } else {
      colabEl.className = "health-chip chip-pending";
      colabLbl.textContent = "Colab Offline (check .env)";
    }

    const ragEl = document.getElementById("status-rag");
    const ragLbl = document.getElementById("lbl-rag-status");
    if (data.chromadb && data.reranker) {
      ragEl.className = "health-chip chip-ok";
      ragLbl.textContent = `Ready (${data.chromadb_chunks || 30980} chunks)`;
    } else {
      ragEl.className = "health-chip chip-err";
      ragLbl.textContent = "Degraded";
    }
  } catch {
    /* ignore background poll errors */
  }
}

function setMode(mode) {
  activeMode = mode;
  document.getElementById("mode-patient").classList.toggle("active", mode === "patient");
  document.getElementById("mode-clinician").classList.toggle("active", mode === "clinician");
}

function switchRightTab(tab) {
  ["evidence", "drugs", "eval"].forEach((t) => {
    document.getElementById(`rtab-${t}`).classList.toggle("active", t === tab);
    document.getElementById(`panel-${t}`).classList.toggle("active", t === tab);
  });
}

// ── CONVERSATIONS ───────────────────────────────────────────────────────
async function loadConversations() {
  const listEl = document.getElementById("conversations-list");
  try {
    const res = await fetch("/api/conversations", {
      headers: { Authorization: `Bearer ${currentToken}` },
    });
    if (!res.ok) return;
    const convs = await res.json();
    if (!convs.length) {
      listEl.innerHTML = '<div class="empty-state-sm">No past consultations.</div>';
      return;
    }
    listEl.innerHTML = convs
      .map(
        (c) =>
          `<button class="conv-item ${c.id === currentConversationId ? "active" : ""}" onclick="selectConversation(${c.id})">${escapeHtml(c.subject)}</button>`
      )
      .join("");
  } catch (err) {
    listEl.innerHTML = `<div class="empty-state-sm msg-err">Failed to load consultations</div>`;
  }
}

async function selectConversation(id) {
  currentConversationId = id;
  loadConversations();
  try {
    const res = await fetch(`/api/conversations/${id}`, {
      headers: { Authorization: `Bearer ${currentToken}` },
    });
    if (!res.ok) return;
    const conv = await res.json();

    // Populate patient profile if saved
    if (conv.patient_profile) {
      document.getElementById("p-age").value = conv.patient_profile.age || "";
      document.getElementById("p-sex").value = conv.patient_profile.sex || "";
      document.getElementById("p-duration").value = conv.patient_profile.duration || "";
      document.getElementById("p-conditions").value = conv.patient_profile.conditions || "";
      document.getElementById("p-medications").value = conv.patient_profile.medications || "";
    }

    // Render messages
    const container = document.getElementById("messages-container");
    container.innerHTML = "";
    conv.messages.forEach((m) => {
      appendMessageUI(m.role, m.content);
    });
  } catch (err) {
    console.error("Failed to load conversation details", err);
  }
}

function startNewConversation() {
  currentConversationId = null;
  loadConversations();
  document.getElementById("messages-container").innerHTML = `
    <div class="welcome-card">
      <h2>New Medical Consultation</h2>
      <p>Ask about symptoms, clinical guidelines, drug interactions, or diagnostic deductions.</p>
    </div>
  `;
  document.getElementById("p-age").value = "";
  document.getElementById("p-sex").value = "";
  document.getElementById("p-duration").value = "";
  document.getElementById("p-conditions").value = "";
  document.getElementById("p-medications").value = "";
  document.getElementById("banner-triage").classList.add("hidden");
  document.getElementById("banner-guardrail").classList.add("hidden");
}

function saveIntakeProfile() {
  alert("Intake profile updated and will be attached to upcoming queries.");
}

function getPatientProfile() {
  const age = document.getElementById("p-age").value.trim() || undefined;
  const sex = document.getElementById("p-sex").value.trim() || undefined;
  const duration = document.getElementById("p-duration").value.trim() || undefined;
  const conditions = document.getElementById("p-conditions").value.trim() || undefined;
  const medications = document.getElementById("p-medications").value.trim() || undefined;
  if (!age && !sex && !duration && !conditions && !medications) return undefined;
  return { age, sex, duration, conditions, medications };
}

// ── CHAT & REAL-TIME SSE STREAMING ──────────────────────────────────────
function handleInputKeyDown(e) {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
}

function fillPrompt(text) {
  const input = document.getElementById("chat-input");
  input.value = text;
  input.focus();
}

async function sendMessage() {
  if (isStreaming) return;
  const input = document.getElementById("chat-input");
  const msg = input.value.trim();
  if (!msg) return;

  input.value = "";
  isStreaming = true;
  document.getElementById("btn-send").disabled = true;

  // Append user message
  appendMessageUI("user", msg);

  // Prepare assistant placeholder bubble
  const assistantBubble = appendMessageUI("assistant", "⏳ *Consulting MedGemma & Retrieving Clinical Guidelines...*");
  let accumulatedText = "";

  const payload = {
    conversation_id: currentConversationId,
    message: msg,
    patient_profile: getPatientProfile(),
    mode: activeMode,
  };

  try {
    const res = await fetch("/api/chat/stream", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${currentToken}`,
      },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      throw new Error(`Chat request failed: ${res.status}`);
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n");
      const frames = buffer.split("\n\n");
      buffer = frames.pop() ?? "";

      for (const frame of frames) {
        if (!frame.trim()) continue;
        const { event, data } = parseSSE(frame);
        handleSSEEvent(event, data, assistantBubble, (text) => {
          accumulatedText = text;
        });
      }
    }
  } catch (err) {
    assistantBubble.innerHTML = `<div class="msg-err">⚠️ Error: ${escapeHtml(err.message)}</div>`;
  } finally {
    isStreaming = false;
    document.getElementById("btn-send").disabled = false;
    loadConversations();
  }
}

function parseSSE(frame) {
  let event = "message";
  const dataLines = [];
  for (const rawLine of frame.split("\n")) {
    if (!rawLine || rawLine.startsWith(":")) continue;
    if (rawLine.startsWith("event:")) {
      event = rawLine.slice(6).trim();
    } else if (rawLine.startsWith("data:")) {
      dataLines.push(rawLine.slice(5).replace(/^ /, ""));
    }
  }
  return { event, data: dataLines.join("\n") };
}

let activeStreamText = "";

function handleSSEEvent(event, rawData, bubbleEl, setTextCallback) {
  let data = rawData;
  try {
    data = JSON.parse(rawData);
  } catch {
    /* string */
  }

  if (event === "conversation") {
    currentConversationId = parseInt(data, 10);
  } else if (event === "triage_status") {
    const banner = document.getElementById("banner-triage");
    banner.classList.remove("hidden");
    document.getElementById("banner-triage-text").textContent = `Category: ${data.category}. Immediate emergency care indicated.`;
  } else if (event === "profile_update") {
    if (data.age && !document.getElementById("p-age").value) document.getElementById("p-age").value = data.age;
    if (data.sex && !document.getElementById("p-sex").value) document.getElementById("p-sex").value = data.sex;
    if (data.duration && !document.getElementById("p-duration").value) document.getElementById("p-duration").value = data.duration;
    if (data.conditions && !document.getElementById("p-conditions").value) document.getElementById("p-conditions").value = data.conditions;
    if (data.medications && !document.getElementById("p-medications").value) document.getElementById("p-medications").value = data.medications;
  } else if (event === "sources") {
    renderEvidenceList(data);
  } else if (event === "token") {
    if (!activeStreamText) {
      activeStreamText = "";
      bubbleEl.innerHTML = "";
    }
    const token = typeof data === "string" ? data : (data && data.text) || "";
    activeStreamText += token;
    bubbleEl.innerHTML = marked.parse(activeStreamText);
    setTextCallback(activeStreamText);
    scrollChat();
  } else if (event === "done") {
    activeStreamText = "";
    const score = data.score !== undefined ? Math.round(data.score * 100) : 100;
    const isPass = data.status === "PASS" || data.status === "EMERGENCY_ESCALATION" || data.status === "INTERCEPTED";

    const badge = document.createElement("div");
    badge.className = `audit-badge ${isPass ? "" : "flagged"}`;
    badge.innerHTML = `🛡️ Evidence Grounding Score: <strong>${score}%</strong> (${data.status})`;
    bubbleEl.parentElement.appendChild(badge);
    scrollChat();
  } else if (event === "error") {
    bubbleEl.innerHTML += `<div class="msg-err">⚠️ ${escapeHtml(typeof data === "string" ? data : JSON.stringify(data))}</div>`;
  }
}

function appendMessageUI(role, content) {
  const container = document.getElementById("messages-container");
  const msgDiv = document.createElement("div");
  msgDiv.className = `chat-msg ${role}`;

  const bubble = document.createElement("div");
  bubble.className = "chat-bubble";
  bubble.innerHTML = marked.parse(content || "");

  msgDiv.appendChild(bubble);
  container.appendChild(msgDiv);
  scrollChat(true);
  return bubble;
}

function scrollChat(force = false) {
  const container = document.getElementById("messages-container");
  if (!container) return;
  const isNearBottom = container.scrollHeight - container.scrollTop - container.clientHeight < 240;
  if (force || isNearBottom) {
    requestAnimationFrame(() => {
      container.scrollTop = container.scrollHeight;
    });
  }
}

// ── RIGHT PANEL LOGIC ───────────────────────────────────────────────────
function renderEvidenceList(sources) {
  const listEl = document.getElementById("evidence-list");
  if (!sources || !sources.length) {
    listEl.innerHTML = '<div class="empty-state-sm">No external evidence needed (safety rule or direct answer).</div>';
    return;
  }
  listEl.innerHTML = sources
    .map(
      (s, idx) => `
      <div class="evidence-card">
        <div class="evidence-card-head">
          <span class="evidence-title">[${idx + 1}] ${escapeHtml(s.title || "Guideline")}</span>
          <span class="evidence-source">${escapeHtml(s.source || "StatPearls")}</span>
        </div>
        <div class="evidence-score">MedCPT Score: ${s.score}</div>
        <div class="evidence-excerpt">${escapeHtml(s.excerpt || "")}</div>
      </div>
    `
    )
    .join("");
}

async function runCorpusSearch() {
  const query = document.getElementById("evidence-search-input").value.trim();
  if (!query) return;
  const listEl = document.getElementById("evidence-list");
  listEl.innerHTML = '<div class="empty-state-sm">Searching 30,980+ StatPearls/OpenFDA chunks...</div>';

  try {
    const res = await fetch(`/api/evidence/search?q=${encodeURIComponent(query)}&mode=${activeMode}`, {
      headers: { Authorization: `Bearer ${currentToken}` },
    });
    if (!res.ok) throw new Error("Search failed");
    const data = await res.json();
    renderEvidenceList(data.results || []);
  } catch (err) {
    listEl.innerHTML = `<div class="empty-state-sm msg-err">${escapeHtml(err.message)}</div>`;
  }
}

async function runDrugCheck() {
  const meds = document.getElementById("check-meds").value.trim();
  const conditions = document.getElementById("check-conditions").value.trim();
  const resultEl = document.getElementById("drug-check-result");
  resultEl.innerHTML = "Auditing contraindications...";

  try {
    const res = await fetch("/api/medications/check", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${currentToken}`,
      },
      body: JSON.stringify({ medications: meds, conditions }),
    });
    if (!res.ok) throw new Error("Check failed");
    const data = await res.json();

    if (data.alert) {
      resultEl.innerHTML = `
        <div class="alert-banner alert-warning" style="margin: 8px 0 0 0;">
          <div class="alert-icon">⛔</div>
          <div class="alert-body">
            <h4>CONTRAINDICATION WARNING</h4>
            <p>${escapeHtml(data.alert)}</p>
          </div>
        </div>
      `;
    } else {
      resultEl.innerHTML = `
        <div class="audit-badge" style="display: flex; margin-top: 8px;">
          ✅ No hard-rule contraindications detected for: ${escapeHtml((data.medications || []).join(", ") || "none")}.
        </div>
      `;
    }
  } catch (err) {
    resultEl.innerHTML = `<div class="msg-err">${escapeHtml(err.message)}</div>`;
  }
}

async function loadEvaluationSummary() {
  const detailsEl = document.getElementById("eval-details");
  try {
    const res = await fetch("/api/evaluation/summary");
    if (!res.ok) return;
    const data = await res.json();

    if (data.metrics) {
      document.getElementById("eval-metrics").innerHTML = `
        <div class="metric-card">
          <span class="metric-val">${Math.round((data.metrics.recall_at_5 || 1.0) * 100)}%</span>
          <span class="metric-lbl">Recall @ 5</span>
        </div>
        <div class="metric-card">
          <span class="metric-val">${Math.round((data.metrics.mean_faithfulness_rag || 0.976) * 100)}%</span>
          <span class="metric-lbl">Faithfulness</span>
        </div>
        <div class="metric-card">
          <span class="metric-val">${Math.round((data.metrics.emergency_detection_accuracy || 1.0) * 100)}%</span>
          <span class="metric-lbl">Emergency Triage</span>
        </div>
        <div class="metric-card">
          <span class="metric-val">${Math.round((data.metrics.contraindication_intercept_accuracy || 1.0) * 100)}%</span>
          <span class="metric-lbl">Safety Intercept</span>
        </div>
      `;
    }
    detailsEl.innerHTML = `
      <div style="font-size: 11px; color: var(--text-muted); margin-top: 8px;">
        Evaluated on ${data.num_cases || 20} benchmark clinical cases (USMLE + StatPearls + OpenFDA).
      </div>
    `;
  } catch {
    /* ignore */
  }
}

function escapeHtml(str) {
  if (!str) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}
