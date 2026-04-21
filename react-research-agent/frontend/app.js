const form = document.getElementById("research-form");
const questionEl = document.getElementById("question");
const maxStepsEl = document.getElementById("maxSteps");
const modelEl = document.getElementById("model");
const submitEl = document.getElementById("submit");
const statusEl = document.getElementById("status");
const reportEl = document.getElementById("report");

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
  reportEl.textContent = "Running research...";
  setStatus("Sending request...");

  try {
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

    reportEl.textContent = body.report || "No report returned.";
    setStatus("Research complete.");
  } catch (error) {
    const message = error instanceof Error ? error.message : "Request failed.";
    reportEl.textContent = "";
    setStatus(message, true);
  } finally {
    submitEl.disabled = false;
  }
});
