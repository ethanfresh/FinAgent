const runBtn = document.getElementById("run-btn");
const turnsInput = document.getElementById("turns-input");
const runStatusEl = document.getElementById("run-status");
const personaGridEl = document.getElementById("persona-grid");
const reportSelectEl = document.getElementById("report-select");
const reportSummaryEl = document.getElementById("report-summary");
const reportViewEl = document.getElementById("report-view");
const fixesListEl = document.getElementById("fixes-list");

let pollHandle = null;

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

// Renders `[title](url)` markdown links as real anchors; everything else is
// HTML-escaped plain text, matching app.js's rendering of the same answer shape.
function renderText(text) {
  const escaped = escapeHtml(text || "");
  const linkPattern = /\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g;
  return escaped.replace(
    linkPattern,
    (match, title, url) => `<a href="${url}" target="_blank" rel="noopener noreferrer">${title}</a>`
  );
}

async function loadPersonas() {
  try {
    const res = await fetch("/api/redteam/personas");
    if (!res.ok) return;
    const personas = await res.json();
    personaGridEl.innerHTML = personas
      .map(
        (p) => `
        <div class="persona-card">
          <div class="persona-name">${escapeHtml(p.name)}</div>
          <div class="persona-goal">${escapeHtml(p.goal)}</div>
        </div>`
      )
      .join("");
  } catch (err) {
    personaGridEl.innerHTML = `<p class="empty-reports">Could not load personas: ${escapeHtml(err.message)}</p>`;
  }
}

function fixHtml(fix) {
  const severity = (fix.severity || "low").toLowerCase();
  const reportLink = fix.verified_by_report
    ? `<button type="button" class="fix-report-link" data-report="${escapeHtml(fix.verified_by_report)}">view verifying report →</button>`
    : "";
  return `
    <div class="session-card">
      <div class="session-head">
        <div class="session-persona">${escapeHtml(fix.title || fix.id || "")}</div>
        <span class="severity ${escapeHtml(severity)}">${escapeHtml(severity)}</span>
      </div>
      <p class="session-goal">${escapeHtml(fix.category || "")}</p>
      ${fix.example_quote ? `<div class="issue-quote">${renderText(fix.example_quote)}</div>` : ""}
      <p class="issue-problem">${renderText(fix.fix_summary || "")}</p>
      ${fix.verification_note ? `<p class="issue-problem">${renderText(fix.verification_note)}</p>` : ""}
      <div class="fix-meta">
        ${fix.fixed_at ? `<span>fixed ${escapeHtml(fix.fixed_at)}</span>` : ""}
        ${reportLink}
      </div>
    </div>`;
}

async function loadFixes() {
  try {
    const res = await fetch("/api/redteam/fixes");
    if (!res.ok) return;
    const fixes = await res.json();
    if (!fixes.length) {
      fixesListEl.innerHTML = '<p class="empty-reports">No fixes recorded yet.</p>';
      return;
    }
    fixesListEl.innerHTML = fixes.map(fixHtml).join("");
    fixesListEl.querySelectorAll(".fix-report-link").forEach((btn) => {
      btn.addEventListener("click", () => {
        reportSelectEl.value = btn.dataset.report;
        loadReport(btn.dataset.report);
        document.getElementById("report-select").scrollIntoView({ behavior: "smooth", block: "center" });
      });
    });
  } catch (err) {
    fixesListEl.innerHTML = `<p class="empty-reports">Could not load fixes: ${escapeHtml(err.message)}</p>`;
  }
}

function issueHtml(issue) {
  const severity = (issue.severity || "low").toLowerCase();
  return `
    <div class="issue">
      <div class="issue-meta">
        <span class="severity ${escapeHtml(severity)}">${escapeHtml(severity)}</span>
        <span class="issue-category">${escapeHtml(issue.category || "")}</span>
        <span class="issue-turn">turn ${escapeHtml(String(issue.turn ?? "?"))}</span>
      </div>
      <p class="issue-problem">${renderText(issue.problem || "")}</p>
      ${issue.quote ? `<div class="issue-quote">${renderText(issue.quote)}</div>` : ""}
    </div>`;
}

function transcriptHtml(transcript) {
  return transcript
    .map(
      (t) => `
      <div class="transcript-turn">
        <div class="transcript-role">User</div>
        <div class="transcript-text">${renderText(t.user)}</div>
      </div>
      <div class="transcript-turn">
        <div class="transcript-role">FinAgent</div>
        <div class="transcript-text">${renderText(t.assistant)}</div>
        <div class="transcript-tools">tools: ${(t.tool_calls || []).map((c) => c.name).join(", ") || "none"}</div>
      </div>`
    )
    .join("");
}

function sessionHtml(session) {
  const issues = session.issues || [];
  const badgeClass = issues.length ? "some" : "zero";
  return `
    <div class="session-card">
      <div class="session-head">
        <div class="session-persona">${escapeHtml(session.persona)}</div>
        <span class="issue-count-badge ${badgeClass}">${issues.length} issue${issues.length === 1 ? "" : "s"}</span>
      </div>
      <p class="session-goal">${escapeHtml(session.goal)}</p>
      ${issues.length ? issues.map(issueHtml).join("") : '<p class="no-issues">No issues flagged.</p>'}
      <details class="transcript-toggle">
        <summary>View transcript</summary>
        ${transcriptHtml(session.transcript || [])}
      </details>
    </div>`;
}

function renderReport(report) {
  const totalIssues = (report.sessions || []).reduce((sum, s) => sum + (s.issues || []).length, 0);
  reportSummaryEl.textContent = `${report.sessions.length} sessions, ${report.turns_per_session} turns each, ${totalIssues} issue(s) flagged against ${report.base_url}`;
  reportViewEl.innerHTML = report.sessions.map(sessionHtml).join("");
}

async function loadReport(name) {
  if (!name) {
    reportViewEl.innerHTML = "";
    reportSummaryEl.textContent = "";
    return;
  }
  try {
    const res = await fetch(`/api/redteam/reports/${encodeURIComponent(name)}`);
    if (!res.ok) return;
    renderReport(await res.json());
  } catch (err) {
    reportViewEl.innerHTML = `<p class="empty-reports">Could not load report: ${escapeHtml(err.message)}</p>`;
  }
}

async function loadReports(selectName) {
  try {
    const res = await fetch("/api/redteam/reports");
    if (!res.ok) return;
    const reports = await res.json();
    if (!reports.length) {
      reportSelectEl.innerHTML = "";
      reportViewEl.innerHTML = '<p class="empty-reports">No red-team runs yet — run one above.</p>';
      reportSummaryEl.textContent = "";
      return;
    }
    reportSelectEl.innerHTML = reports
      .map((r) => `<option value="${escapeHtml(r.name)}">${escapeHtml(r.name)} — ${r.issues} issue(s), ${r.sessions} sessions</option>`)
      .join("");
    const target = selectName && reports.some((r) => r.name === selectName) ? selectName : reports[0].name;
    reportSelectEl.value = target;
    await loadReport(target);
  } catch (err) {
    reportViewEl.innerHTML = `<p class="empty-reports">Could not load reports: ${escapeHtml(err.message)}</p>`;
  }
}

const DEMO_NOTICE = "Live runs are disabled on the public demo — browse the saved reports below.";
let runEnabled = true;

function setRunning(isRunning, label) {
  runBtn.disabled = isRunning || !runEnabled;
  turnsInput.disabled = isRunning || !runEnabled;
  runStatusEl.textContent = label || (runEnabled ? "" : DEMO_NOTICE);
  runStatusEl.className = "run-status" + (isRunning ? " running" : "");
}

// A run costs real API budget, so the hosted demo serves saved reports only.
// Reflect that in the controls instead of letting the button fire a request
// the server is going to reject with a 403.
async function applyConfig() {
  try {
    const res = await fetch("/api/config");
    if (!res.ok) return;
    const config = await res.json();
    runEnabled = config.redteam_run_enabled !== false;
  } catch {
    // Leave the controls enabled — the server still enforces the real rule.
  }
  setRunning(false, "");
}

async function pollStatus() {
  try {
    const res = await fetch("/api/redteam/status");
    if (!res.ok) return;
    const status = await res.json();

    if (status.status === "running") {
      setRunning(true, `Running — ${status.turns} turns per persona, this takes a few minutes...`);
      if (!pollHandle) pollHandle = setInterval(pollStatus, 3000);
      return;
    }

    clearInterval(pollHandle);
    pollHandle = null;

    if (status.status === "error") {
      setRunning(false, `Last run failed: ${status.error}`);
    } else if (status.status === "done") {
      setRunning(false, "Done.");
      await loadReports();
    } else {
      setRunning(false, "");
    }
  } catch (err) {
    clearInterval(pollHandle);
    pollHandle = null;
    setRunning(false, `Lost contact with the server: ${err.message}`);
  }
}

async function startRun() {
  if (!runEnabled) return;
  const turns = Math.max(1, Math.min(8, parseInt(turnsInput.value, 10) || 4));
  setRunning(true, "Starting...");
  try {
    const res = await fetch("/api/redteam/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ turns }),
    });
    const data = await res.json();
    if (!res.ok) {
      // The server is the authority on whether runs are allowed; if it says no,
      // stop offering the button regardless of what /api/config reported.
      if (res.status === 403) runEnabled = false;
      setRunning(false, data.detail || `Could not start run (HTTP ${res.status}).`);
      return;
    }
    if (data.status === "already_running") {
      setRunning(true, "A run is already in progress...");
    }
    if (!pollHandle) pollHandle = setInterval(pollStatus, 3000);
    pollStatus();
  } catch (err) {
    setRunning(false, `Could not start run: ${err.message}`);
  }
}

runBtn.addEventListener("click", startRun);
reportSelectEl.addEventListener("change", () => loadReport(reportSelectEl.value));

// applyConfig first, so the run controls settle into the right state before
// pollStatus can report on them.
applyConfig().then(pollStatus);
loadPersonas();
loadFixes();
loadReports();
