const form = document.getElementById("research-form");
const questionEl = document.getElementById("question");
const maxStepsEl = document.getElementById("maxSteps");
const modelEl = document.getElementById("model");
const submitEl = document.getElementById("submit");
const statusEl = document.getElementById("status");
const reportEl = document.getElementById("report");
const agentStateBadgeEl = document.getElementById("agentStateBadge");
const agentToolTextEl = document.getElementById("agentToolText");

function getApiBaseUrl() {
  if (window.APP_CONFIG && window.APP_CONFIG.API_BASE_URL) {
    return String(window.APP_CONFIG.API_BASE_URL).replace(/\/+$/, "");
  }
  return "http://localhost:8000";
}

function setStatus(message, isError = false) {
  statusEl.textContent = message;
  statusEl.classList.toggle("error", Boolean(isError));
}

function sleep(ms) {
  return new Promise((resolve) => {
    setTimeout(resolve, ms);
  });
}

function toStateLabel(state) {
  const labels = {
    idle: "Idle",
    queued: "Queued",
    thinking: "Thinking",
    using_tool: "Using Tool",
    finalizing: "Finalizing",
    completed: "Completed",
    failed: "Failed",
  };
  return labels[state] || "Working";
}

function setAssistantState(state, tool = "") {
  const normalized = state || "idle";
  agentStateBadgeEl.textContent = toStateLabel(normalized);
  agentStateBadgeEl.className = `state-badge state-${normalized}`;
  agentToolTextEl.textContent = tool ? `Tool: ${tool}` : "";
}

function escapeHtml(text) {
  return String(text)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function renderReport(markdownText) {
  const lines = String(markdownText || "").split(/\r?\n/);
  const html = [];

  for (const line of lines) {
    const trimmed = line.trimEnd();
    if (!trimmed) {
      html.push('<p>&nbsp;</p>');
      continue;
    }

    if (trimmed.startsWith("## ")) {
      html.push(`<h2 class="report-h2"><strong>${escapeHtml(trimmed.slice(3))}</strong></h2>`);
      continue;
    }

    if (trimmed.startsWith("# ")) {
      html.push(`<h1 class="report-h1"><strong>${escapeHtml(trimmed.slice(2))}</strong></h1>`);
      continue;
    }

    html.push(`<p>${escapeHtml(trimmed)}</p>`);
  }

  reportEl.innerHTML = html.join("");
}

async function runResearchSync(payload) {
  setAssistantState("thinking");
  const response = await fetch(`${getApiBaseUrl()}/research`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  const body = await response.json();
  if (!response.ok) {
    const detail = body && body.detail ? body.detail : `HTTP ${response.status}`;
    throw new Error(detail);
  }

  renderReport(body.report || "No report returned.");
  setAssistantState("completed");
  setStatus("Research complete.");
}

async function createResearchJob(payload) {
  const response = await fetch(`${getApiBaseUrl()}/research/jobs`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (response.status === 404) {
    return null;
  }

  const body = await response.json();
  if (!response.ok) {
    const detail = body && body.detail ? body.detail : `HTTP ${response.status}`;
    throw new Error(detail);
  }

  return body.job_id;
}

async function pollResearchJob(jobId) {
  for (let attempt = 0; attempt < 600; attempt += 1) {
    const response = await fetch(`${getApiBaseUrl()}/research/jobs/${jobId}`);
    const body = await response.json();
    if (!response.ok) {
      const detail = body && body.detail ? body.detail : `HTTP ${response.status}`;
      throw new Error(detail);
    }

    setAssistantState(body.state, body.tool || "");
    if (body.message) {
      setStatus(body.message, body.state === "failed");
    }

    if (body.done) {
      if (body.state === "completed") {
        renderReport(body.report || "No report returned.");
        setStatus("Research complete.");
      } else {
        renderReport("");
        setStatus(body.error || body.message || "Research failed.", true);
      }
      return;
    }

    await sleep(1200);
  }

  throw new Error("Timed out waiting for research job.");
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const question = questionEl.value.trim();
  if (question.length < 3) {
    setStatus("Question must be at least 3 characters.", true);
    return;
  }

  const payload = { question };
  const maxStepsValue = Number(maxStepsEl.value);
  if (Number.isFinite(maxStepsValue)) {
    payload.max_steps = Math.min(30, Math.max(1, maxStepsValue));
  }

  const model = modelEl.value.trim();
  if (model) {
    payload.model = model;
  }

  submitEl.disabled = true;
  setAssistantState("queued");
  renderReport("Running research...");
  setStatus("Submitting request...");

  try {
    const jobId = await createResearchJob(payload);
    if (jobId) {
      await pollResearchJob(jobId);
    } else {
      await runResearchSync(payload);
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : "Request failed.";
    setAssistantState("failed");
    renderReport("");
    setStatus(message, true);
  } finally {
    submitEl.disabled = false;
  }
});
