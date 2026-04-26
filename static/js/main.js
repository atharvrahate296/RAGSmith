/**
 * RAGSmith – Frontend SPA v2.0
 * Pure Vanilla JS — no framework dependencies
 * New in v2.0: Hybrid search scores, re-rank display, confidence badges, grounding bars
 */

'use strict';

// ── State ──────────────────────────────────────────────────────────────────
const state = {
  projects: [],
  activeProject: null,
  documents: [],
  chatHistory: [],
  pollTimers: {},
};

// ── API Helpers ────────────────────────────────────────────────────────────
async function api(method, path, body = null, isForm = false) {
  const opts = { method, headers: {} };
  if (body && !isForm) {
    opts.headers['Content-Type'] = 'application/json';
    opts.body = JSON.stringify(body);
  } else if (body && isForm) {
    opts.body = body; // FormData
  }
  const res = await fetch('/api' + path, opts);
  if (res.status === 204) return null;
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
  return data;
}

// ── Toast ──────────────────────────────────────────────────────────────────
function toast(msg, type = 'info') {
  const el = document.getElementById('toast');
  el.textContent = msg;
  el.className = `toast show toast-${type}`;
  clearTimeout(el._t);
  el._t = setTimeout(() => { el.className = 'toast'; }, 3500);
}

// ── Views ──────────────────────────────────────────────────────────────────
function showView(name) {
  document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
  document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('view-' + name)?.classList.add('active');
  document.querySelector(`[data-view="${name}"]`)?.classList.add('active');

  if (name === 'documents') loadDocuments();
  if (name === 'history')   loadHistory();
}

function enableProjectViews(project) {
  state.activeProject = project;
  ['documents', 'query', 'history'].forEach(v => {
    const btn = document.getElementById('nav-' + v);
    if (btn) btn.disabled = false;
  });
  document.getElementById('active-project-label').textContent =
    '◈  ' + project.name;
  document.getElementById('doc-project-name').textContent   = '/ ' + project.name;
  document.getElementById('query-project-name').textContent = '/ ' + project.name;
  document.getElementById('history-project-name').textContent = '/ ' + project.name;
}

// ── Projects ───────────────────────────────────────────────────────────────
async function loadProjects() {
  try {
    state.projects = await api('GET', '/projects/');
    renderProjects();
  } catch (e) {
    toast('Failed to load projects: ' + e.message, 'error');
  }
}

function renderProjects() {
  const grid = document.getElementById('projects-grid');
  if (!state.projects.length) {
    grid.innerHTML = `
      <div class="empty-state">
        <div class="empty-state-icon">◈</div>
        <p>No projects yet. Create your first RAG project.</p>
      </div>`;
    return;
  }

  grid.innerHTML = state.projects.map(p => `
    <div class="project-card ${state.activeProject?.id === p.id ? 'selected' : ''}"
         onclick="selectProject(${p.id})">
      <div class="project-card-header">
        <div class="project-name">${escHtml(p.name)}</div>
        <div class="project-actions">
          <button class="btn-icon" title="Delete project"
            onclick="event.stopPropagation(); deleteProject(${p.id}, '${escHtml(p.name)}')">🗑</button>
        </div>
      </div>
      <div class="project-desc">${escHtml(p.description) || '<span style="opacity:.4">No description</span>'}</div>
      <div class="project-meta">
        <span class="badge badge-accent">${escHtml(p.model)}</span>
        <span class="badge">top-k: ${p.top_k}</span>
        <span class="badge">${p.document_count} doc${p.document_count !== 1 ? 's' : ''}</span>
        <span class="badge" style="margin-left:auto;">
          <a href="/api/export/${p.id}" style="color:inherit;text-decoration:none" title="Download export">↓ export</a>
        </span>
      </div>
    </div>
  `).join('');
}

async function selectProject(id) {
  const project = state.projects.find(p => p.id === id);
  if (!project) return;
  enableProjectViews(project);
  // Reset chat
  state.chatHistory = [];
  renderChat();
  renderProjects();
  showView('documents');
}

async function createProject() {
  const name = document.getElementById('new-project-name').value.trim();
  const desc = document.getElementById('new-project-desc').value.trim();
  const model = document.getElementById('new-project-model').value;
  const topK = parseInt(document.getElementById('new-project-topk').value, 10);

  if (!name) { toast('Project name is required.', 'warn'); return; }

  try {
    await api('POST', '/projects/', { name, description: desc, model, top_k: topK });
    closeModal('modal-new-project');
    document.getElementById('new-project-name').value = '';
    document.getElementById('new-project-desc').value = '';
    toast('Project created ✓', 'success');
    await loadProjects();
  } catch (e) {
    toast(e.message, 'error');
  }
}

async function deleteProject(id, name) {
  if (!confirm(`Delete project "${name}" and all its documents? This cannot be undone.`)) return;
  try {
    await api('DELETE', `/projects/${id}`);
    if (state.activeProject?.id === id) {
      state.activeProject = null;
      ['documents', 'query', 'history'].forEach(v => {
        const btn = document.getElementById('nav-' + v);
        if (btn) btn.disabled = true;
      });
      document.getElementById('active-project-label').textContent = 'No project selected';
      showView('projects');
    }
    toast('Project deleted', 'success');
    await loadProjects();
  } catch (e) {
    toast(e.message, 'error');
  }
}

// ── Documents ──────────────────────────────────────────────────────────────
async function loadDocuments() {
  if (!state.activeProject) return;
  try {
    state.documents = await api('GET', `/documents/${state.activeProject.id}`);
    renderDocuments();
    // Poll for processing docs
    const processing = state.documents.filter(d => ['pending','processing'].includes(d.status));
    if (processing.length) scheduleDocPoll();
  } catch (e) {
    toast('Failed to load documents: ' + e.message, 'error');
  }
}

function docIcon(filename) {
  const ext = filename.split('.').pop().toLowerCase();
  const map = { pdf: '📄', txt: '📝', md: '📋', docx: '📃', doc: '📃', csv: '📊', rst: '📜' };
  return map[ext] || '📄';
}

function renderDocuments() {
  const list = document.getElementById('doc-list');
  if (!state.documents.length) {
    list.innerHTML = `
      <div class="empty-state">
        <div class="empty-state-icon">◉</div>
        <p>No documents yet. Upload a file to build your knowledge base.</p>
      </div>`;
    return;
  }

  list.innerHTML = state.documents.map(d => `
    <div class="doc-item">
      <span class="doc-icon">${docIcon(d.filename)}</span>
      <div class="doc-info">
        <div class="doc-name">${escHtml(d.filename)}</div>
        <div class="doc-meta">
          ${d.num_chunks ? d.num_chunks + ' chunks · ' : ''}
          ${d.created_at.split('T')[0]}
          ${d.error_msg ? ' · <span style="color:var(--danger)">' + escHtml(d.error_msg.substring(0,80)) + '</span>' : ''}
        </div>
      </div>
      <span class="doc-status status-${d.status}">${d.status}</span>
      ${d.status === 'error' ? `<button class="btn btn-ghost" style="font-size:11px;padding:5px 10px;" title="Retry processing" onclick="retryDocument(${d.id})">↺ Retry</button>` : ''}
      <button class="btn-icon" title="Delete" onclick="deleteDocument(${d.id})">🗑</button>
    </div>
  `).join('');
}

function scheduleDocPoll() {
  const pid = state.activeProject?.id;
  if (!pid) return;
  clearTimeout(state.pollTimers[pid]);
  state.pollTimers[pid] = setTimeout(async () => {
    await loadDocuments();
  }, 3000);
}

async function uploadDocument(input) {
  if (!state.activeProject || !input.files.length) return;
  const file = input.files[0];
  input.value = '';

  const fd = new FormData();
  fd.append('file', file);

  try {
    await api('POST', `/documents/${state.activeProject.id}/upload`, fd, true);
    toast(`Uploading "${file.name}"…`, 'info');
    await loadDocuments();
    scheduleDocPoll();
  } catch (e) {
    toast('Upload failed: ' + e.message, 'error');
  }
}

async function deleteDocument(docId) {
  if (!confirm('Delete this document?')) return;
  try {
    await api('DELETE', `/documents/${state.activeProject.id}/doc/${docId}`);
    toast('Document deleted', 'success');
    await loadDocuments();
  } catch (e) {
    toast(e.message, 'error');
  }
}

async function retryDocument(docId) {
  try {
    await api('POST', `/documents/${state.activeProject.id}/doc/${docId}/retry`);
    toast('Retrying document processing…', 'info');
    await loadDocuments();
    scheduleDocPoll();
  } catch (e) {
    toast('Retry failed: ' + e.message, 'error');
  }
}

// ── Query / Chat ───────────────────────────────────────────────────────────
function handleQueryKey(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendQuery();
  }
}

async function sendQuery() {
  if (!state.activeProject) return;
  const input = document.getElementById('query-input');
  const query = input.value.trim();
  if (!query) return;

  input.value = '';
  const btn = document.getElementById('query-send-btn');
  btn.disabled = true;

  // Add user bubble
  state.chatHistory.push({ role: 'user', content: query });
  renderChat();

  // Add thinking indicator
  const thinkingId = 'thinking-' + Date.now();
  appendThinking(thinkingId);

  try {
    const res = await api('POST', `/query/${state.activeProject.id}`, { query });
    removeThinking(thinkingId);
    state.chatHistory.push({
      role: 'ai',
      content: res.answer,
      sources: res.sources,
      model: res.model,
      grounding_score: res.grounding_score ?? 0,
      query_relevance: res.query_relevance ?? 0,
      confidence_label: res.confidence_label ?? 'low',
    });
    renderChat();
  } catch (e) {
    removeThinking(thinkingId);
    state.chatHistory.push({ role: 'ai', content: '⚠ ' + e.message, sources: [] });
    renderChat();
    toast(e.message, 'error');
  } finally {
    btn.disabled = false;
    input.focus();
  }
}

// ── Confidence helpers ─────────────────────────────────────────────────────
function confidenceEmoji(label) {
  return { high: '🟢', medium: '🟡', low: '🔴' }[label] ?? '⚪';
}

function clamp01(v) { return Math.min(1, Math.max(0, v || 0)); }

function confidenceBadgeHtml(label, grounding, relevance) {
  const emoji = confidenceEmoji(label);
  const cls   = `confidence-${label}`;
  const gPct  = (clamp01(grounding) * 100).toFixed(1);
  const rPct  = (clamp01(relevance) * 100).toFixed(1);
  const gW    = (clamp01(grounding) * 100).toFixed(1);
  const rW    = (clamp01(relevance) * 100).toFixed(1);

  return `
    <div class="confidence-block">
      <span class="confidence-badge ${cls}">${emoji} ${label} confidence</span>
    </div>
    <div class="score-bars">
      <div class="score-bar-row">
        <span class="score-bar-label">Grounding</span>
        <div class="score-bar-track">
          <div class="score-bar-fill grounding" style="width:${gW}%"></div>
        </div>
        <span class="score-bar-value">${gPct}%</span>
      </div>
      <div class="score-bar-row">
        <span class="score-bar-label">Relevance</span>
        <div class="score-bar-track">
          <div class="score-bar-fill relevance" style="width:${rW}%"></div>
        </div>
        <span class="score-bar-value">${rPct}%</span>
      </div>
    </div>`;
}

function sourceItemHtml(s, i) {
  const isTop  = s.is_top_source;
  const rankChange = s.original_rank - i;  // positive = jumped up
  let rankHtml = '';
  if (s.original_rank !== undefined && s.original_rank !== i) {
    if (rankChange > 0) {
      rankHtml = `<div class="source-rank-change improved">↑ jumped from rank #${s.original_rank + 1} → #${i + 1} after re-ranking</div>`;
    } else {
      rankHtml = `<div class="source-rank-change dropped">↓ moved from rank #${s.original_rank + 1} → #${i + 1}</div>`;
    }
  }

  return `
    <div class="source-item${isTop ? ' top-source' : ''}">
      <div class="source-file">📄 ${escHtml(s.doc_filename)}</div>
      <div class="source-scores">
        <span class="source-score-pill">Dense <span>${(s.dense_score || 0).toFixed(4)}</span></span>
        <span class="source-score-pill">BM25 <span>${(s.bm25_score || 0).toFixed(4)}</span></span>
        <span class="source-score-pill">RRF <span>${(s.rrf_score || 0).toFixed(6)}</span></span>
        <span class="source-score-pill">Rerank <span>${(s.rerank_score || 0).toFixed(4)}</span></span>
      </div>
      ${rankHtml}
      <div class="source-preview">${escHtml(s.text.substring(0, 200))}…</div>
    </div>`;
}

// ── Render Chat ────────────────────────────────────────────────────────────
function renderChat() {
  const container = document.getElementById('chat-container');
  const empty = document.getElementById('chat-empty');

  if (!state.chatHistory.length) {
    empty.style.display = 'flex';
    Array.from(container.children)
      .filter(c => c !== empty && !c.classList.contains('thinking'))
      .forEach(c => c.remove());
    return;
  }

  empty.style.display = 'none';
  Array.from(container.children)
    .filter(c => c !== empty && !c.id?.startsWith('thinking'))
    .forEach(c => c.remove());

  state.chatHistory.forEach((msg, i) => {
    const el = document.createElement('div');
    el.className = `chat-msg chat-msg-${msg.role}`;
    el.dataset.idx = i;

    if (msg.role === 'user') {
      el.innerHTML = `
        <div class="chat-msg-label">YOU</div>
        <div class="bubble">${escHtml(msg.content)}</div>`;
    } else {
      // Confidence + grounding bars (only when we have evaluation data)
      const hasEval = msg.confidence_label && msg.confidence_label !== 'low' ||
                      msg.grounding_score > 0 || msg.query_relevance > 0;
      const evalHtml = (msg.sources?.length && hasEval !== false && msg.confidence_label)
        ? confidenceBadgeHtml(msg.confidence_label, msg.grounding_score, msg.query_relevance)
        : '';

      // Sources dropdown
      const sourcesHtml = msg.sources?.length ? `
        <button class="sources-toggle" onclick="toggleSources(this)">
          ▸ ${msg.sources.length} source${msg.sources.length > 1 ? 's' : ''} · ${escHtml(msg.model || '')}
        </button>
        <div class="sources-list">
          ${msg.sources.map((s, si) => sourceItemHtml(s, si)).join('')}
        </div>` : '';

      el.innerHTML = `
        <div class="chat-msg-label">RAGSMITH</div>
        <div class="bubble">${escHtml(msg.content)}</div>
        ${evalHtml}
        ${sourcesHtml}`;
    }
    container.appendChild(el);
  });

  container.scrollTop = container.scrollHeight;
}

function toggleSources(btn) {
  const list = btn.nextElementSibling;
  list.classList.toggle('open');
  btn.textContent = btn.textContent.startsWith('▸')
    ? btn.textContent.replace('▸', '▾')
    : btn.textContent.replace('▾', '▸');
}

function appendThinking(id) {
  const container = document.getElementById('chat-container');
  document.getElementById('chat-empty').style.display = 'none';
  const el = document.createElement('div');
  el.id = id;
  el.className = 'chat-msg chat-msg-ai thinking';
  el.innerHTML = `
    <div class="chat-msg-label">RAGSMITH</div>
    <div class="thinking-bubble">
      <div class="dot"></div><div class="dot"></div><div class="dot"></div>
    </div>`;
  container.appendChild(el);
  container.scrollTop = container.scrollHeight;
}

function removeThinking(id) {
  document.getElementById(id)?.remove();
}

// ── History ────────────────────────────────────────────────────────────────
async function loadHistory() {
  if (!state.activeProject) return;
  try {
    const logs = await api('GET', `/query/${state.activeProject.id}/history?limit=50`);
    const list = document.getElementById('history-list');
    if (!logs.length) {
      list.innerHTML = `<div class="empty-state"><div class="empty-state-icon">◷</div><p>No queries yet.</p></div>`;
      return;
    }
    list.innerHTML = logs.map(l => {
      const label = l.grounding_score >= 0.75 ? 'high' : l.grounding_score >= 0.50 ? 'medium' : 'low';
      const emoji = confidenceEmoji(label);
      const badgeHtml = l.grounding_score > 0
        ? `<span class="history-confidence confidence-${label}">${emoji} ${(l.grounding_score * 100).toFixed(0)}%</span>`
        : '';
      return `
        <div class="history-item">
          <div class="history-query">▸ ${escHtml(l.query_text)} ${badgeHtml}</div>
          <div class="history-answer">${escHtml(l.response)}</div>
          <div class="history-meta">${l.created_at} · ${l.num_chunks} chunks retrieved</div>
        </div>`;
    }).join('');
  } catch (e) {
    toast('Failed to load history: ' + e.message, 'error');
  }
}

// ── Modal ──────────────────────────────────────────────────────────────────
function openModal(id) {
  document.getElementById(id)?.classList.add('open');
}

function closeModal(id) {
  document.getElementById(id)?.classList.remove('open');
}

function closeModalOutside(e, id) {
  if (e.target === document.getElementById(id)) closeModal(id);
}

// ── LLM Status ────────────────────────────────────────────────────────────────
async function checkOllama() {
  const el = document.getElementById('ollama-status');
  try {
    const res = await fetch('/health', { signal: AbortSignal.timeout(5000) });
    if (res.ok) {
      const data = await res.json();
      const provider = (data.llm_provider || 'llm').toUpperCase();
      if (data.llm_available) {
        el.innerHTML = `<span class="status-dot running"></span> ${provider} running`;
      } else {
        el.innerHTML = `<span class="status-dot stopped"></span> ${provider} offline`;
        el.title = data.llm_detail || '';
      }
    } else {
      throw new Error();
    }
  } catch {
    el.innerHTML = '<span class="status-dot stopped"></span> LLM offline';
  }
}

// ── Utils ──────────────────────────────────────────────────────────────────
function escHtml(str) {
  if (!str) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

// ── Init ───────────────────────────────────────────────────────────────────
(async function init() {
  await loadProjects();
  await checkOllama();
  setInterval(checkOllama, 30_000);
})();
