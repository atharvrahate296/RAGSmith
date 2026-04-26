/**
 * RAGSmith – Frontend SPA v2.1
 * Pure Vanilla JS — no framework dependencies
 * v2.1: Dynamic model switching, Docs view, improved error handling & loading states.
 */

'use strict';

// ── State ──────────────────────────────────────────────────────────────────
const state = {
  projects: [],
  activeProject: null,
  documents: [],
  activeSession: null,
  chatHistory: [], 
  chatSessions: [],
  availableModels: { groq: [], ollama: [] }, // Initialize as object
  pollTimers: {},
  showRetrievalDetails: false,
};

// ── API Helpers ────────────────────────────────────────────────────────────
async function api(method, path, body = null, isForm = false) {
  const opts = { method, headers: {} };
  if (body && !isForm) {
    opts.headers['Content-Type'] = 'application/json';
    opts.body = JSON.stringify(body);
  } else if (body && isForm) {
    opts.body = body; 
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
  if (name === 'query') {
    if (state.activeSession) {
      loadChatHistory(state.activeSession.id);
    } else {
      state.chatHistory = [];
      renderChat();
    }
  }
}

function enableProjectViews(project) {
  state.activeProject = project;
  ['documents', 'query'].forEach(v => {
    const btn = document.getElementById('nav-' + v);
    if (btn) btn.disabled = false;
  });
  document.getElementById('active-project-label').textContent = '◈ ' + project.name;
  document.getElementById('doc-project-name').textContent   = '/ ' + project.name;
  document.getElementById('query-project-name').textContent = '/ ' + project.name;
  document.getElementById('new-chat-btn').disabled = false;
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
          <button class="btn-icon" title="Rename project" onclick="event.stopPropagation(); renameProject(${p.id}, '${escHtml(p.name)}')">✎</button>
          <button class="btn-icon" title="Delete project"
            onclick="event.stopPropagation(); deleteProject(${p.id}, '${escHtml(p.name)}')">🗑</button>
        </div>
      </div>
      <div class="project-desc">${escHtml(p.description) || '<span style="opacity:.4">No description</span>'}</div>
      <div class="project-meta">
        <span class="badge badge-secondary">${escHtml(p.provider)}</span>
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
  
  state.activeSession = null;
  state.chatHistory = [];
  renderChat();
  renderProjects(); 
  showView('query'); 
  loadChatSessions(id); 
}

async function createProject() {
  const name = document.getElementById('new-project-name').value.trim();
  const desc = document.getElementById('new-project-desc').value.trim();
  const provider = document.querySelector('input[name="project-provider"]:checked').value;
  const model = document.getElementById('new-project-model').value;
  const topK = parseInt(document.getElementById('new-project-topk').value, 10);

  if (!name) {
    toast('Project name is required.', 'warn');
    return;
  }

  const createBtn = document.querySelector('#modal-new-project .btn-primary');
  const originalText = createBtn.textContent;
  createBtn.textContent = 'Creating...';
  createBtn.disabled = true;

  try {
    await api('POST', '/projects/', { name, description: desc, provider, model, top_k: topK });
    closeModal('modal-new-project');
    document.getElementById('new-project-name').value = '';
    document.getElementById('new-project-desc').value = '';
    toast('Project created ✓', 'success');
    await loadProjects();
  } catch (e) {
    toast(e.message, 'error');
  } finally {
    createBtn.textContent = originalText;
    createBtn.disabled = false;
  }
}

async function renameProject(projectId, currentName) {
  const newName = prompt('Enter new project name:', currentName);
  if (!newName || newName === currentName) return;
  try {
    await api('PATCH', `/projects/${projectId}`, { name: newName });
    toast('Project renamed ✓', 'success');
    await loadProjects();
    if (state.activeProject?.id === projectId) {
        document.getElementById('active-project-label').textContent = '◈  ' + newName;
    }
  } catch (e) {
    toast(e.message, 'error');
  }
}

async function deleteProject(id, name) {
  if (!confirm(`Delete project "${name}" and all its documents and chat sessions? This cannot be undone.`)) return;
  try {
    await api('DELETE', `/projects/${id}`);
    if (state.activeProject?.id === id) {
      state.activeProject = null;
      state.activeSession = null;
      state.chatSessions = [];
      state.chatHistory = [];
      renderChatSessions();
      renderChat();
      ['documents', 'query', 'history'].forEach(v => {
        const btn = document.getElementById('nav-' + v);
        if (btn) btn.disabled = true;
      });
      document.getElementById('new-chat-btn').disabled = true;
      document.getElementById('active-project-label').textContent = 'No project selected';
      showView('projects');
    }
    toast('Project deleted', 'success');
    await loadProjects();
  } catch (e) {
    toast(e.message, 'error');
  }
}

// ── Chat Sessions ──────────────────────────────────────────────────────────
async function loadChatSessions(projectId) {
  try {
    state.chatSessions = await api('GET', `/sessions/project/${projectId}`);
    renderChatSessions();
    if (!state.activeSession && state.chatSessions.length > 0) {
      selectChatSession(state.chatSessions[0].id);
    } else if (state.activeSession && !state.chatSessions.find(s => s.id === state.activeSession.id)) {
      state.activeSession = null;
      state.chatHistory = [];
      renderChat();
    }
  } catch (e) {
    toast('Failed to load chat sessions: ' + e.message, 'error');
  }
}

function renderChatSessions() {
  const list = document.getElementById('chat-sessions-list');
  if (!state.chatSessions.length) {
    list.innerHTML = `
      <div class="sessions-empty">
        <span>No chats yet</span>
      </div>`;
    return;
  }

  list.innerHTML = state.chatSessions.map(s => {
    const isActive = state.activeSession?.id === s.id;
    const date = new Date(s.created_at);
    const now = new Date();
    const isToday = date.toDateString() === now.toDateString();
    const dateStr = isToday
      ? date.toLocaleTimeString(undefined, { hour: 'numeric', minute: '2-digit' })
      : date.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
    const modelShort = (s.model || '').split(':')[0].replace('llama-', 'llama').replace('-instant','').slice(0, 16);

    return `
      <div class="session-item ${isActive ? 'active' : ''}" onclick="selectChatSession(${s.id})" title="${escHtml(s.name)}">
        <div class="session-item-indicator"></div>
        <div class="session-item-body">
          <div class="session-item-name">${escHtml(s.name)}</div>
          <div class="session-item-meta">
            <span class="session-model-chip">${escHtml(modelShort)}</span>
            <span class="session-date">${dateStr}</span>
          </div>
        </div>
        <div class="session-item-actions">
          <button class="session-action-btn" title="Settings" onclick="event.stopPropagation(); openChatSettings(${s.id})">⚙</button>
          <button class="session-action-btn danger" title="Delete" onclick="event.stopPropagation(); deleteChatSession(${s.id}, '${escHtml(s.name)}')">✕</button>
        </div>
      </div>`;
  }).join('');
}

async function selectChatSession(sessionId) {
  const session = state.chatSessions.find(s => s.id === sessionId);
  if (!session) return;
  state.activeSession = session;
  renderChatSessions(); 
  document.getElementById('query-project-name').textContent = `/ ${state.activeProject.name} / ${session.name}`;
  loadChatHistory(sessionId);
}

async function newChat() {
  if (!state.activeProject) {
    toast('Please select a project first.', 'warn');
    showView('projects');
    return;
  }
  const newChatBtn = document.getElementById('new-chat-btn');
  const originalText = newChatBtn.textContent;
  newChatBtn.textContent = 'Creating...';
  newChatBtn.disabled = true;

  try {
    const newName = new Date().toLocaleString(undefined, { month: 'numeric', day: 'numeric', year: 'numeric', hour: 'numeric', minute: 'numeric', second: 'numeric' });
    const session = await api('POST', '/sessions/', {
      project_id: state.activeProject.id,
      name: newName,
      provider: state.activeProject.provider,
      model: state.activeProject.model, 
    });
    state.chatSessions.unshift(session);
    renderChatSessions();
    selectChatSession(session.id);
    showView('query');
    toast('New chat started ✓', 'success');
  } catch (e) {
    toast('Failed to create new chat: ' + e.message, 'error');
  } finally {
    newChatBtn.textContent = originalText;
    newChatBtn.disabled = false;
  }
}

async function openChatSettings(sessionId) {
  const session = state.chatSessions.find(s => s.id === sessionId);
  if (!session) return;

  document.getElementById('chat-settings-name').value = session.name;
  const modelSelect = document.getElementById('chat-settings-model');
  modelSelect.innerHTML = ''; 

  const provider = session.provider || state.activeProject.provider || 'ollama';
  const models = state.availableModels[provider] || [];

  if (models.length === 0) {
    modelSelect.innerHTML = '<option value="">No models available</option>';
    modelSelect.disabled = true;
  } else {
    modelSelect.disabled = false;
    models.forEach(modelName => {
      const opt = document.createElement('option');
      opt.value = modelName;
      opt.textContent = modelName;
      modelSelect.appendChild(opt);
    });
  }
  modelSelect.value = session.model || state.activeProject.model;
  modelSelect.dataset.sessionId = sessionId; 
  openModal('modal-chat-settings');
}

async function saveChatSettings() {
  const name = document.getElementById('chat-settings-name').value.trim();
  const model = document.getElementById('chat-settings-model').value;
  const sessionId = parseInt(document.getElementById('chat-settings-model').dataset.sessionId, 10);

  if (!name) {
    toast('Session name is required.', 'warn');
    return;
  }

  const saveBtn = document.querySelector('#modal-chat-settings .btn-primary');
  const originalText = saveBtn.textContent;
  saveBtn.textContent = 'Saving...';
  saveBtn.disabled = true;

  try {
    const currentSession = state.chatSessions.find(s => s.id === sessionId);
    const provider = currentSession.provider; // Current provider is implicitly used if not changed via UI
    const updatedSession = await api('PATCH', `/sessions/${sessionId}`, { name, provider, model });
    toast('Chat settings saved ✓', 'success');
    closeModal('modal-chat-settings');
    
    const idx = state.chatSessions.findIndex(s => s.id === sessionId);
    if (idx !== -1) {
      state.chatSessions[idx] = updatedSession;
    }
    if (state.activeSession?.id === sessionId) {
      state.activeSession = updatedSession;
      document.getElementById('query-project-name').textContent = `/ ${state.activeProject.name} / ${updatedSession.name}`;
    }
    renderChatSessions();
  } catch (e) {
    toast(e.message, 'error');
  } finally {
    saveBtn.textContent = originalText;
    saveBtn.disabled = false;
  }
}

async function deleteChatSession(sessionId, name) {
  if (!confirm(`Delete chat session "${name}"? This cannot be undone.`)) return;
  try {
    await api('DELETE', `/sessions/${sessionId}`);
    toast('Chat session deleted', 'success');
    state.chatSessions = state.chatSessions.filter(s => s.id !== sessionId);
    renderChatSessions();
    if (state.activeSession?.id === sessionId) {
      state.activeSession = null;
      state.chatHistory = [];
      renderChat();
      document.getElementById('query-project-name').textContent = `/ ${state.activeProject.name}`;
    }
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

function pipelineAnimate(phase) {
  const ids = ['ps-hybrid', 'ps-rerank', 'ps-eval'];
  ids.forEach((id, i) => {
    const el = document.getElementById(id);
    if (!el) return;
    el.classList.toggle('stage-active', phase > 0 && i + 1 <= phase && phase < 4);
  });
}

async function sendQuery() {
  if (!state.activeProject || !state.activeSession) {
    toast('Please select a project and a chat session first.', 'warn');
    return;
  }
  const input = document.getElementById('query-input');
  const query = input.value.trim();
  if (!query) return;

  input.value = '';
  const btn = document.getElementById('query-send-btn');
  btn.disabled = true;

  state.chatHistory.push({ role: 'user', content: query });
  renderChat();

  const thinkingId = 'thinking-' + Date.now();
  appendThinking(thinkingId);
  pipelineAnimate(1);
  const t2 = setTimeout(() => pipelineAnimate(2), 400);
  const t3 = setTimeout(() => pipelineAnimate(3), 800);

  try {
    const res = await api('POST', `/query/${state.activeProject.id}`, {
      query,
      session_id: state.activeSession.id,
      model: state.activeSession.model || state.activeProject.model,
    });
    clearTimeout(t2); clearTimeout(t3);
    pipelineAnimate(4);
    setTimeout(() => pipelineAnimate(0), 1500);
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
    
    const idx = state.chatSessions.findIndex(s => s.id === state.activeSession.id);
    if (idx !== -1) {
      state.chatSessions[idx].updated_at = new Date().toISOString();
      state.chatSessions.sort((a, b) => new Date(b.updated_at) - new Date(a.updated_at));
      renderChatSessions();
    }
  } catch (e) {
    clearTimeout(t2); clearTimeout(t3);
    pipelineAnimate(0);
    removeThinking(thinkingId);
    state.chatHistory.push({ role: 'ai', content: '⚠ ' + e.message, sources: [] });
    renderChat();
    toast(e.message, 'error');
  } finally {
    btn.disabled = false;
    input.focus();
  }
}

async function loadChatHistory(sessionId) {
  try {
    const logs = await api('GET', `/sessions/${sessionId}/history`);
    state.chatHistory = logs.map(l => [
      { role: 'user', content: l.query_text },
      { 
        role: 'ai', 
        content: l.response, 
        sources: [], 
        model: l.model,
        grounding_score: l.grounding_score ?? 0,
        query_relevance: l.query_relevance ?? 0,
        confidence_label: l.confidence_label ?? 'low'
      }
    ]).flat();
    renderChat();
  } catch (e) {
    toast('Failed to load chat history: ' + e.message, 'error');
    state.chatHistory = [];
    renderChat();
  }
}

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

    if (msg.role === 'user') {
      el.innerHTML = `
        <div class="chat-msg-label">YOU</div>
        <div class="bubble">${escHtml(msg.content)}</div>`;
    } else {
      const hasScores = (msg.grounding_score ?? 0) > 0 || (msg.query_relevance ?? 0) > 0;
      const evalHtml = hasScores
        ? confidenceBadgeHtml(msg.confidence_label ?? 'low', msg.grounding_score ?? 0, msg.query_relevance ?? 0)
        : '';
      const capturedCardSeq = _cardSeq;

      const sourcesHtml = msg.sources?.length ? `
        <div class="sources-header">
          <button class="sources-toggle" onclick="toggleSources(this)">
            ▸ ${msg.sources.length} source${msg.sources.length > 1 ? 's' : ''} · ${escHtml(msg.provider || 'unknown')} / ${escHtml(msg.model || '')}
          </button>
          
        </div>
        <div class="sources-list">
          ${msg.sources.map((s, si) => sourceItemHtml(s, si)).join('')}
        </div>` : '';

      el.innerHTML = `
        <div class="chat-msg-label">RAGSMITH</div>
        <div class="bubble">${escHtml(msg.content)}</div>
        ${evalHtml}
        ${sourcesHtml}`;

      if (hasScores) {
        animateConfCard(capturedCardSeq, (msg.grounding_score ?? 0) * 100, (msg.query_relevance ?? 0) * 100);
      }
    }
    container.appendChild(el);
  });
  container.scrollTop = container.scrollHeight;
}

function toggleRetrievalDetails(event) {
  state.showRetrievalDetails = event.target.checked;
  renderChat();
}

function toggleSources(btn) {
  const list = btn.parentElement.nextElementSibling;
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

// ── Confidence helpers ─────────────────────────────────────────────────────
function confidenceEmoji(label) {
  return { high: '🟢', medium: '🟡', low: '🔴' }[label] ?? '⚪';
}
function clamp01(v) { return Math.min(1, Math.max(0, v || 0)); }

let _cardSeq = 0;
function confidenceBadgeHtml(label, grounding, relevance) {
  const id    = ++_cardSeq;
  const emoji = confidenceEmoji(label);
  const cls   = `confidence-${label}`;
  const gPct  = (clamp01(grounding) * 100).toFixed(1);
  const rPct  = (clamp01(relevance) * 100).toFixed(1);
  const gaugeColour = { high: '#2ed573', medium: '#ffa502', low: '#ff4757' }[label] ?? '#535d6b';
  const gDeg  = Math.round(clamp01(grounding) * 360);
  const gaugeStyle = `background: conic-gradient(${gaugeColour} ${gDeg}deg, #1a1f28 ${gDeg}deg)`;

  return `
    <div class="confidence-card card-${label}" id="conf-card-${id}">
      <div class="conf-gauge">
        <div class="conf-gauge-ring" style="${gaugeStyle}">
          <span class="conf-gauge-val">${gPct}<span style="font-size:9px">%</span></span>
        </div>
        <span class="conf-gauge-label">${label}</span>
      </div>
      <div class="conf-scores">
        <div class="conf-badge-row">
          <span class="confidence-badge ${cls}">${emoji} ${label} confidence</span>
        </div>
        <div class="score-bars">
          <div class="score-bar-row">
            <span class="score-bar-label">Grounding</span>
            <div class="score-bar-track">
              <div class="score-bar-fill grounding" id="gbar-${id}" style="width:0%"></div>
            </div>
            <span class="score-bar-value">${gPct}%</span>
          </div>
          <div class="score-bar-row">
            <span class="score-bar-label">Relevance</span>
            <div class="score-bar-track">
              <div class="score-bar-fill relevance" id="rbar-${id}" style="width:0%"></div>
            </div>
            <span class="score-bar-value">${rPct}%</span>
          </div>
        </div>
      </div>
    </div>`;
}

function animateConfCard(id, gPct, rPct) {
  requestAnimationFrame(() => {
    setTimeout(() => {
      const gb = document.getElementById('gbar-' + id);
      const rb = document.getElementById('rbar-' + id);
      if (gb) gb.style.width = gPct + '%';
      if (rb) rb.style.width = rPct + '%';
    }, 60);
  });
}

function sourceItemHtml(s, i) {
  const isTop = s.is_top_source;
  const rankChange = s.original_rank - i;
  let rankHtml = '';
  if (s.original_rank !== undefined && s.original_rank !== i) {
    if (rankChange > 0) {
      rankHtml = `<div class="source-rank-change improved">↑ ${s.original_rank + 1} → ${i + 1} after re-rank</div>`;
    } else {
      rankHtml = `<div class="source-rank-change dropped">↓ ${s.original_rank + 1} → ${i + 1}</div>`;
    }
  }
  const scoreDetailsHtml = `
    <div class="source-scores">
      <span class="source-score-pill pill-dense">Dense <span>${(s.dense_score || 0).toFixed(3)}</span></span>
      <span class="source-score-pill pill-bm25">BM25 <span>${(s.bm25_score || 0).toFixed(3)}</span></span>
      <span class="source-score-pill pill-rrf">RRF <span>${(s.rrf_score || 0).toFixed(5)}</span></span>
      <span class="source-score-pill pill-rerank">Rerank <span>${(s.rerank_score || 0).toFixed(3)}</span></span>
    </div>`;
  return `
    <div class="source-item${isTop ? ' top-source' : ''}">
      <span class="source-rank-num">#${i + 1}</span>
      <div class="source-file">📄 ${escHtml(s.doc_filename)}</div>
      ${scoreDetailsHtml}
      ${rankHtml}
      <div class="source-preview">${escHtml(s.text.substring(0, 220))}…</div>
    </div>`;
}

// ── Settings ───────────────────────────────────────────────────────────────
async function checkOllama() {
  const el = document.getElementById('ollama-status');
  try {
    const res = await fetch('/health');
    const data = await res.json();
    const provider = (data.llm_provider || 'LLM').toUpperCase();
    if (data.llm_available) {
      el.innerHTML = `<span class="status-dot running"></span> ${provider} running`;
    } else {
      el.innerHTML = `<span class="status-dot stopped"></span> ${provider} offline`;
    }
  } catch {
    el.innerHTML = '<span class="status-dot stopped"></span> LLM offline';
  }
}

async function loadSettingsModal() {
  try {
    const keyStatus = await api('GET', '/settings/groq/key-status');
    const keyInput = document.getElementById('groq-api-key');
    if (keyStatus.configured) keyInput.placeholder = '••••••••••••••••••••••••';
    
    const groqModels = state.availableModels.groq || [];
    const ollamaModels = state.availableModels.ollama || [];

    const groqModelsList = document.getElementById('groq-models-list');
    const ollamaModelsList = document.getElementById('ollama-models-list');

    groqModelsList.innerHTML = '';
    if (groqModels.length) {
      groqModels.forEach(m => groqModelsList.innerHTML += `<div class="model-item">${escHtml(m)}</div>`);
    } else {
      groqModelsList.innerHTML = '<p style="opacity:.6;">No Groq models available</p>';
    }

    ollamaModelsList.innerHTML = '';
    if (ollamaModels.length) {
      ollamaModels.forEach(m => ollamaModelsList.innerHTML += `<div class="model-item">${escHtml(m)}</div>`);
    } else {
      ollamaModelsList.innerHTML = '<p style="opacity:.6;">No Ollama models available</p>';
    }
    
    const settings = await api('GET', '/settings/');
    document.getElementById('llm-provider-info').textContent = settings.llm_provider.toUpperCase();
    document.getElementById('active-model-info').textContent = state.activeProject?.model || settings.available_models[0] || 'Not set'; // This part might need further refinement based on actual setting structure
  } catch (e) { console.error('Settings load failed', e); }
}

async function testGroqKey() {
  const apiKey = document.getElementById('groq-api-key').value.trim();
  if (!apiKey) { toast('Enter a Groq API key.', 'warn'); return; }
  const btn = event.target;
  btn.disabled = true; btn.textContent = 'Testing...';
  try {
    const res = await api('POST', '/settings/groq/validate', { api_key: apiKey });
    toast(res.valid ? 'Valid key!' : 'Invalid key: ' + res.message, res.valid ? 'success' : 'error');
  } catch (e) { toast(e.message, 'error'); }
  finally { btn.disabled = false; btn.textContent = 'Test Key'; }
}

async function saveGroqSettings() {
  const apiKey = document.getElementById('groq-api-key').value.trim();
  if (!apiKey) { toast('Enter a Groq API key.', 'warn'); return; }
  const btn = event.target;
  btn.disabled = true; btn.textContent = 'Saving...';
  try {
    await api('POST', '/settings/groq/save', { api_key: apiKey });
    toast('Settings saved', 'success');
    closeModal('modal-app-settings');
    const models = await api('GET', '/settings/models');
    state.availableModels = models.models || [];
  } catch (e) { toast(e.message, 'error'); }
  finally { btn.disabled = false; btn.textContent = 'Save Settings'; }
}

// ── Utils ──────────────────────────────────────────────────────────────────
function escHtml(str) {
  if (!str) return '';
  return String(str).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}
function openModal(id) {
  document.getElementById(id)?.classList.add('open');
  if (id === 'modal-app-settings') loadSettingsModal();
}
function closeModal(id) { document.getElementById(id)?.classList.remove('open'); }
function closeModalOutside(e, id) { if (e.target === document.getElementById(id)) closeModal(id); }

// ── Init ───────────────────────────────────────────────────────────────────
(async function init() {
  await loadProjects();
  await checkOllama();
  setInterval(checkOllama, 30000);
  
  // Load models for both providers
  await loadAvailableModels('groq');
  await loadAvailableModels('ollama');
  
  // Initialize project creation modal with default provider
  updateProjectModels('groq'); 
})();

async function loadAvailableModels(provider) {
  try {
    const models = await api('GET', `/settings/models?provider=${provider}`);
    state.availableModels[provider] = models.models || [];
  } catch (e) {
    console.error(`Failed to load ${provider} models:`, e);
    state.availableModels[provider] = [`${provider} models unavailable`];
  }
}

function updateProjectModels(provider) {
  const modelSelect = document.getElementById('new-project-model');
  modelSelect.innerHTML = '';
  const models = state.availableModels[provider] || [];
  if (models.length === 0) {
    modelSelect.innerHTML = '<option value="">No models available</option>';
    modelSelect.disabled = true;
    return;
  }
  modelSelect.disabled = false;
  models.forEach(m => {
    const opt = document.createElement('option');
    opt.value = m;
    opt.textContent = m;
    modelSelect.appendChild(opt);
  });
  // Pre-select a default if available
  if (models.length > 0) {
    modelSelect.value = models[0];
  }
}