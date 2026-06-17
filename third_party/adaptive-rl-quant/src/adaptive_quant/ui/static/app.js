const readinessPill = document.getElementById("readiness-pill");
const readinessDetail = document.getElementById("readiness-detail");
const statusGrid = document.getElementById("status-grid");
const actionsEl = document.getElementById("actions");
const logOutput = document.getElementById("log-output");
const activeJobLabel = document.getElementById("active-job-label");
const jobList = document.getElementById("job-list");
const gpuPanel = document.getElementById("gpu-panel");
const outputsPanel = document.getElementById("outputs-panel");
const repoRoot = document.getElementById("repo-root");
const gitLine = document.getElementById("git-line");
const refreshBtn = document.getElementById("refresh-btn");
const workflowSelect = document.getElementById("workflow-select");
const workflowDescription = document.getElementById("workflow-description");
const workflowPresets = document.getElementById("workflow-presets");
const workflowFields = document.getElementById("workflow-fields");
const environmentFields = document.getElementById("environment-fields");
const commandPreview = document.getElementById("command-preview");
const runBtn = document.getElementById("run-btn");
const chatBackend = document.getElementById("chat-backend");
const chatModel = document.getElementById("chat-model");
const chatHardware = document.getElementById("chat-hardware");
const chatDownloadBtn = document.getElementById("chat-download-model-btn");
const chatHfSearch = document.getElementById("chat-hf-search");
const chatHfSearchBtn = document.getElementById("chat-hf-search-btn");
const chatHfResults = document.getElementById("chat-hf-results");
const chatLog = document.getElementById("chat-log");
const chatForm = document.getElementById("chat-form");
const chatInput = document.getElementById("chat-input");
const chatMeta = document.getElementById("chat-meta");
const chatAppendTask = document.getElementById("chat-append-task");
const chatResetBtn = document.getElementById("chat-reset-btn");
const chatRunContinuousBtn = document.getElementById("chat-run-continuous-btn");

let catalog = null;
let chatConfig = null;
let statusData = null;
let activeJobId = null;
let pollTimer = null;
let launcherToken = "";

function apiHeaders() {
  const headers = { "Content-Type": "application/json" };
  if (launcherToken) {
    headers["X-Launcher-Token"] = launcherToken;
  }
  return headers;
}

function readinessCopy(readiness) {
  if (readiness === "ready") {
    return ["Ready to run", "Choose a workflow, configure options, and press Run."];
  }
  if (readiness === "warning") {
    return ["Ready with warnings", "Repo is on a slow WSL mount. Consider moving it under ~/src."];
  }
  return ["Setup incomplete", "Run ./setup.sh from the repo root, then refresh this page."];
}

function card(label, value, sub = "") {
  return `
    <article class="status-card">
      <div class="label">${label}</div>
      <div class="value">${value}</div>
      ${sub ? `<div class="sub">${sub}</div>` : ""}
    </article>
  `;
}

function infoRow(label, value) {
  return `<div class="info-row"><span>${label}</span><span>${value}</span></div>`;
}

function currentWorkflow() {
  return catalog?.workflows?.find((item) => item.id === workflowSelect.value) || null;
}

function renderField(field, configFiles, scope = "") {
  const id = `field-${scope}${field.name}`.replace(/[^a-zA-Z0-9_-]/g, "_");
  const help = field.help ? `<small class="muted">${field.help}</small>` : "";
  const privileged = field.privileged ? " field-privileged" : "";

  if (field.type === "choice") {
    const options = (field.choices || [])
      .map(
        (choice) =>
          `<option value="${choice.value}" ${choice.value === field.default ? "selected" : ""}>${choice.label}</option>`,
      )
      .join("");
    return `
      <label class="field${privileged}" for="${id}">
        <span>${field.label}</span>
        <select id="${id}" data-name="${field.name}" data-type="choice">${options}</select>
        ${help}
      </label>
    `;
  }

  if (field.type === "config_file") {
    const fileOptions = (configFiles || [])
      .map((file) => `<option value="${file.path}">${file.label}</option>`)
      .join("");
    return `
      <label class="field${privileged}" for="${id}">
        <span>${field.label}</span>
        <input list="${id}-list" id="${id}" data-name="${field.name}" data-type="text" value="${field.default || ""}" placeholder="config.e2e_smoke.json" />
        <datalist id="${id}-list">${fileOptions}</datalist>
        ${help}
      </label>
    `;
  }

  if (field.type === "lines") {
    return `
      <label class="field field-wide${privileged}" for="${id}">
        <span>${field.label}</span>
        <textarea id="${id}" data-name="${field.name}" data-type="lines" rows="3" placeholder="${field.placeholder || ""}">${field.default || ""}</textarea>
        ${help}
      </label>
    `;
  }

  if (field.type === "checkbox") {
    return `
      <label class="field field-check${privileged}" for="${id}">
        <input type="checkbox" id="${id}" data-name="${field.name}" data-type="checkbox" />
        <span>${field.label}</span>
        ${help}
      </label>
    `;
  }

  const inputType = field.type === "int" || field.type === "float" ? "number" : "text";
  const stepAttr = field.step ? `step="${field.step}"` : field.type === "float" ? 'step="any"' : "";
  const maxAttr = field.max != null ? `max="${field.max}"` : "";
  return `
    <label class="field${privileged}" for="${id}">
      <span>${field.label}</span>
      <input type="${inputType}" id="${id}" data-name="${field.name}" data-type="${field.type}" value="${field.default || ""}" placeholder="${field.placeholder || ""}" ${field.min != null ? `min="${field.min}"` : ""} ${maxAttr} ${stepAttr} />
      ${help}
    </label>
  `;
}

function renderFormFields(container, fields, configFiles, scope = "") {
  container.innerHTML = (fields || []).map((field) => renderField(field, configFiles, scope)).join("");
}

function renderFieldGroups(container, groups, configFiles) {
  const blocks = (groups || []).map((group) => {
    const privilegedNote = group.privileged
      ? `<p class="muted privileged-note">Requires privileged overrides in environment settings.</p>`
      : "";
    const fieldsHtml = (group.fields || [])
      .map((field) => renderField(field, configFiles, `${group.id}-`))
      .join("");
    return `
      <details class="field-group" open>
        <summary>${group.title}</summary>
        ${group.description ? `<p class="muted group-description">${group.description}</p>` : ""}
        ${privilegedNote}
        <div class="form-grid group-fields">${fieldsHtml}</div>
      </details>
    `;
  });
  container.innerHTML = blocks.join("");
}

function collectOptionsFrom(container) {
  const options = {};
  container.querySelectorAll("[data-name]").forEach((element) => {
    const name = element.dataset.name;
    const type = element.dataset.type;
    if (type === "checkbox") {
      if (element.checked) {
        options[name] = true;
      }
      return;
    }
    const value = element.value?.trim();
    if (!value) {
      return;
    }
    if (type === "int" || type === "float") {
      options[name] = Number(value);
      return;
    }
    if (type === "lines") {
      options[name] = value.split("\n").map((line) => line.trim()).filter(Boolean);
      return;
    }
    options[name] = value;
  });
  return options;
}

function collectRunPayload() {
  const workflow = workflowSelect.value;
  const options = {
    ...collectOptionsFrom(workflowFields),
    ...collectOptionsFrom(document.getElementById("workflow-field-groups")),
    ...collectOptionsFrom(environmentFields),
  };
  return { workflow, options };
}

function applyPresetOptions(options) {
  for (const [key, value] of Object.entries(options)) {
    const element = document.querySelector(`[data-name="${key}"]`);
    if (!element) {
      continue;
    }
    if (element.dataset.type === "checkbox") {
      element.checked = Boolean(value);
      continue;
    }
    if (Array.isArray(value)) {
      element.value = value.join("\n");
      continue;
    }
    element.value = value;
  }
  updateCommandPreview();
}

function renderWorkflowForm() {
  const workflow = currentWorkflow();
  if (!workflow) {
    return;
  }
  workflowDescription.textContent = workflow.description || "";
  renderFormFields(workflowFields, workflow.fields, catalog?.config_files || []);
  const groupsEl = document.getElementById("workflow-field-groups");
  if (groupsEl) {
    renderFieldGroups(groupsEl, workflow.field_groups, catalog?.config_files || []);
  }

  workflowPresets.innerHTML = (workflow.presets || [])
    .map(
      (preset, index) =>
        `<button class="ghost preset-btn" type="button" data-preset-index="${index}">${preset.label}</button>`,
    )
    .join("");

  workflowPresets.querySelectorAll("[data-preset-index]").forEach((button) => {
    button.addEventListener("click", () => {
      const preset = workflow.presets[Number(button.dataset.presetIndex)];
      applyPresetOptions(preset.options || {});
    });
  });

  if (catalog?.recommended_nvidia_ack) {
    const ackField = environmentFields.querySelector('[data-name="nvidia_ack"]');
    if (ackField && !ackField.value) {
      ackField.value = catalog.recommended_nvidia_ack;
    }
  }

  updateCommandPreview();
}

function renderEnvironmentForm() {
  renderFormFields(environmentFields, catalog?.environment_fields || [], catalog?.config_files || []);
}

function renderWorkflowSelect() {
  const groups = {};
  for (const workflow of catalog?.workflows || []) {
    groups[workflow.category] = groups[workflow.category] || [];
    groups[workflow.category].push(workflow);
  }
  workflowSelect.innerHTML = Object.entries(groups)
    .map(([category, workflows]) => {
      const options = workflows
        .map((workflow) => `<option value="${workflow.id}">${workflow.title}</option>`)
        .join("");
      return `<optgroup label="${category}">${options}</optgroup>`;
    })
    .join("");
}

async function updateCommandPreview() {
  const payload = collectRunPayload();
  try {
    const response = await fetch("/api/preview", {
      method: "POST",
      headers: apiHeaders(),
      body: JSON.stringify(payload),
    });
    const preview = await response.json();
    if (response.ok) {
      commandPreview.textContent = preview.command_preview || (preview.command || []).join(" ");
      return;
    }
  } catch (_error) {
    // ignore preview failures
  }
  commandPreview.textContent = `${payload.workflow}`;
}

function renderStatus(data) {
  statusData = data;
  const [pillText, detailText] = readinessCopy(data.readiness);
  readinessPill.textContent = pillText;
  readinessPill.className = `status-pill ${data.readiness}`;
  readinessDetail.textContent = detailText;

  const setup = data.setup || {};
  const hardware = data.hardware || {};
  const torch = data.torch || {};
  const platform = data.platform || {};

  statusGrid.innerHTML = [
    card("Environment", setup.package_importable ? "Installed" : "Missing", setup.package_version || "adaptive-rl-quant"),
    card("Hardware", hardware.accelerator_name || "CPU simulator", hardware.accelerator_type || "cpu"),
    card("PyTorch", torch.torch_installed ? (torch.cuda_available ? "CUDA ready" : "CPU wheel") : "Not installed", torch.torch_version || "Optional for GPU"),
    card("Platform", platform.system || "—", platform.wsl2 ? "WSL2" : platform.machine || ""),
  ].join("");

  const nvidia = data.nvidia || {};
  const boundary = nvidia.boundary || {};
  gpuPanel.innerHTML = [
    infoRow("NVIDIA host", nvidia.linux_nvidia_host ? "Yes" : "No"),
    infoRow("nvidia-smi", torch.nvidia_smi_visible ? "Visible" : "Not detected"),
    infoRow("nvidia-smi path", torch.nvidia_smi_path || "—"),
    infoRow("CUDA device", torch.device_name || "—"),
    infoRow("Secure boundary", nvidia.approved_tier || (nvidia.needs_ack_for_gpu_training ? "Ack required" : "N/A")),
    infoRow("WSL2", boundary.wsl2 ? "Yes" : "No"),
  ].join("");

  const outputs = data.outputs || {};
  const outputRows = Object.keys(outputs).length
    ? Object.entries(outputs).map(([name, count]) => infoRow(name, count == null ? "—" : `${count} files`)).join("")
    : infoRow("outputs/", "No runs yet");
  outputsPanel.innerHTML = outputRows;

  repoRoot.textContent = data.repo_root || "—";
  gitLine.textContent = data.git?.head ? `git ${data.git.head}${data.git.dirty ? " (dirty)" : ""}` : "git unavailable";

  actionsEl.innerHTML = (data.actions || [])
    .map((action) => {
      const disabled = !action.enabled;
      const note = action.requires_ack
        ? "Uses host-venv ack from environment settings."
        : action.description;
      return `
        <button
          class="action-card ${action.primary ? "primary" : ""}"
          data-workflow="${action.workflow || action.id}"
          data-options='${JSON.stringify(action.options || {})}'
          ${disabled ? "disabled" : ""}
          type="button"
        >
          <h3>${action.title}</h3>
          <p>${note}</p>
          <span class="tag">${action.category}</span>
        </button>
      `;
    })
    .join("");

  actionsEl.querySelectorAll("[data-workflow]").forEach((button) => {
    button.addEventListener("click", () => {
      workflowSelect.value = button.dataset.workflow;
      renderWorkflowForm();
      applyPresetOptions(JSON.parse(button.dataset.options || "{}"));
      startConfiguredRun();
    });
  });
}

async function fetchCatalog() {
  const response = await fetch("/api/catalog");
  catalog = await response.json();
  launcherToken = catalog.launcher_token || "";
  renderWorkflowSelect();
  renderEnvironmentForm();
  renderWorkflowForm();
  if (catalog?.status) {
    renderStatus(catalog.status);
  }
}

async function fetchStatus() {
  const response = await fetch("/api/status");
  const data = await response.json();
  renderStatus(data);
}

async function fetchJobs() {
  const response = await fetch("/api/jobs");
  const jobs = await response.json();
  jobList.innerHTML = jobs.length
    ? jobs
        .slice()
        .reverse()
        .map(
          (job) => `
            <li>
              <strong>${job.label}</strong>
              <div class="state ${job.status}">${job.status}</div>
            </li>
          `,
        )
        .join("")
    : "<li>No jobs yet.</li>";
  return jobs;
}

async function pollJob(jobId) {
  const response = await fetch(`/api/jobs/${jobId}`);
  const job = await response.json();
  logOutput.textContent = (job.log_tail || []).join("\n") || "Running…";
  logOutput.scrollTop = logOutput.scrollHeight;
  activeJobLabel.textContent = `${job.label} · ${job.status}`;

  if (job.status === "running" || job.status === "queued") {
    pollTimer = window.setTimeout(() => pollJob(jobId), 800);
    return;
  }

  activeJobId = null;
  await fetchJobs();
  await fetchStatus();
}

async function startConfiguredRun() {
  if (activeJobId) {
    logOutput.textContent = "Another job is still running. Wait for it to finish.";
    return;
  }

  const payload = collectRunPayload();
  const workflow = currentWorkflow();
  const label = workflow?.title || payload.workflow;

  logOutput.textContent = `Starting ${label}…`;
  activeJobLabel.textContent = `${label} · starting`;

  const response = await fetch("/api/run", {
    method: "POST",
    headers: apiHeaders(),
    body: JSON.stringify(payload),
  });

  const job = await response.json();
  if (!response.ok) {
    logOutput.textContent = job.error || "Failed to start job.";
    activeJobLabel.textContent = "Start failed";
    return;
  }

  commandPreview.textContent = (job.command || []).join(" ");
  activeJobId = job.job_id;
  if (pollTimer) {
    window.clearTimeout(pollTimer);
  }
  await fetchJobs();
  pollJob(job.job_id);
}

workflowSelect.addEventListener("change", renderWorkflowForm);
workflowFields.addEventListener("input", updateCommandPreview);
document.getElementById("workflow-field-groups")?.addEventListener("input", updateCommandPreview);
environmentFields.addEventListener("input", updateCommandPreview);
runBtn.addEventListener("click", startConfiguredRun);

refreshBtn.addEventListener("click", () => {
  fetchCatalog();
  fetchJobs();
});

fetchCatalog();
fetchJobs();
fetchChatConfig();

function renderChatMessage(role, text, meta = "") {
  const article = document.createElement("article");
  article.className = `chat-bubble chat-${role}`;
  article.innerHTML = `<div class="chat-text">${escapeHtml(text)}</div>${meta ? `<div class="chat-sub muted">${escapeHtml(meta)}</div>` : ""}`;
  chatLog.appendChild(article);
  chatLog.scrollTop = chatLog.scrollHeight;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

async function fetchChatConfig() {
  const response = await fetch("/api/chat/config");
  chatConfig = await response.json();
  chatBackend.innerHTML = (chatConfig.backends || [])
    .filter((item) => item.available !== false)
    .map(
      (item) =>
        `<option value="${item.id}" ${item.id === chatConfig.default_backend ? "selected" : ""}>${item.label}</option>`,
    )
    .join("");


  renderModelOptions(chatConfig.models || [], chatConfig.selected_model_id || "");

  const hf = chatConfig.huggingface_cli ? " · hf CLI ready" : " · install hf CLI to download routes";
  const binary = chatConfig.llama_cpp_binary ? " · llama.cpp binary found" : "";
  chatMeta.textContent = `${chatConfig.chat_tasks_count || 0} queued tasks · ${chatConfig.chat_tasks_path}${hf}${binary}`;
}

function renderModelOptions(models, selectedId) {
  if (!chatModel) {
    return;
  }
  const ready = models.filter((item) => item.ready);
  const pending = models.filter((item) => !item.ready);
  chatModel.innerHTML = [
    `<option value="">Simulator only</option>`,
    ...(ready.length
      ? [`<optgroup label="Ready">${ready.map(modelOptionHtml(selectedId)).join("")}</optgroup>`]
      : []),
    ...(pending.length
      ? [`<optgroup label="Download from Hugging Face">${pending.map(modelOptionHtml(selectedId)).join("")}</optgroup>`]
      : []),
  ].join("");
  if (selectedId) {
    chatModel.value = selectedId;
  }
}

function modelOptionHtml(selectedId) {
  return (item) => {
    const suffix = item.ready ? "" : " (download required)";
    const selected = item.id === selectedId ? "selected" : "";
    return `<option value="${item.id}" ${selected}>${item.label}${suffix}</option>`;
  };
}

async function selectModel(modelId) {
  const response = await fetch("/api/models", {
    method: "POST",
    headers: apiHeaders(),
    body: JSON.stringify({ action: "select", model_id: modelId || null }),
  });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || "Model selection failed");
  }
  chatConfig = { ...chatConfig, ...payload };
  renderModelOptions(payload.models || [], payload.selected_model_id || "");
  return payload;
}

async function downloadSelectedModel() {
  const modelId = chatModel?.value;
  if (!modelId) {
    renderChatMessage("system", "Select a Hugging Face route to download.");
    return;
  }
  chatDownloadBtn.disabled = true;
  chatMeta.textContent = "Downloading GGUF from Hugging Face…";
  try {
    const response = await fetch("/api/models", {
      method: "POST",
      headers: apiHeaders(),
      body: JSON.stringify({ action: "download", model_id: modelId }),
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error || "Download failed");
    }
    chatConfig = { ...chatConfig, ...payload };
    renderModelOptions(payload.models || [], payload.selected_model_id || modelId);
    if (chatBackend.querySelector('option[value="llama_cpp_rl"]')) {
      chatBackend.value = "llama_cpp_rl";
    }
    renderChatMessage("system", `Downloaded ${payload.downloaded?.label || modelId}. Ready for llama.cpp RL.`);
    await fetchChatConfig();
  } catch (error) {
    renderChatMessage("system", error.message || "Download failed.");
  } finally {
    chatDownloadBtn.disabled = false;
  }
}

async function searchHuggingFaceRepos() {
  const query = chatHfSearch?.value?.trim();
  if (!query) {
    return;
  }
  chatHfResults.innerHTML = "<li>Searching…</li>";
  const response = await fetch("/api/models", {
    method: "POST",
    headers: apiHeaders(),
    body: JSON.stringify({ action: "search", query }),
  });
  const payload = await response.json();
  if (!response.ok) {
    chatHfResults.innerHTML = `<li>${payload.error || "Search failed"}</li>`;
    return;
  }
  const rows = payload.results || [];
  chatHfResults.innerHTML = rows.length
    ? rows
        .map(
          (item) =>
            `<li><a href="https://huggingface.co/${item.repo_id}" target="_blank" rel="noopener">${item.repo_id}</a></li>`,
        )
        .join("")
    : "<li>No GGUF repos found. Register routes with adaptive-rl-quant-route register.</li>";
}

async function sendChatMessage(event) {
  event.preventDefault();
  const message = chatInput.value.trim();
  if (!message) {
    return;
  }
  renderChatMessage("user", message);
  chatInput.value = "";
  chatInput.disabled = true;

  const response = await fetch("/api/chat", {
    method: "POST",
    headers: apiHeaders(),
    body: JSON.stringify({
      message,
      backend: chatBackend.value,
      model_id: chatModel?.value || null,
      hardware: chatHardware.value,
      append_task: chatAppendTask.checked,
    }),
  });
  const payload = await response.json();
  chatInput.disabled = false;

  if (!response.ok) {
    renderChatMessage("system", payload.error || "Chat request failed.");
    return;
  }

  const metaParts = [];
  if (payload.reward != null) {
    metaParts.push(`reward=${Number(payload.reward).toFixed(4)}`);
  }
  if (payload.learn_applied) {
    metaParts.push("policy updated");
  }
  if (payload.metrics) {
    metaParts.push(`latency=${Number(payload.metrics.latency_ms || 0).toFixed(1)}ms`);
  }
  renderChatMessage("assistant", payload.response_text || "(empty)", metaParts.join(" · "));
  await fetchChatConfig();
}

async function resetChatSession() {
  const response = await fetch("/api/chat", {
    method: "POST",
    headers: apiHeaders(),
    body: JSON.stringify({ action: "reset_session" }),
  });
  const payload = await response.json();
  if (response.ok) {
    chatLog.innerHTML = "";
    renderChatMessage("system", "Chat RL session reset.");
    await fetchChatConfig();
  } else {
    renderChatMessage("system", payload.error || "Reset failed.");
  }
}

function prefillContinuousFromChat() {
  workflowSelect.value = "continuous";
  renderWorkflowForm();
  const options = {
    config: "config.e2e_smoke.json",
    continuous_task_stream_mode: "jsonl",
    continuous_task_jsonl_path: chatConfig?.chat_tasks_path || "outputs/chat_tasks.jsonl",
  };
  const selected = (chatConfig?.models || []).find((item) => item.id === chatModel?.value);
  if (selected?.ready && selected.model_path) {
    options.backend = "llama_cpp";
    options.llama_cpp_model = selected.model_path;
    if (chatConfig?.llama_cpp_binary) {
      options.llama_cpp_binary = chatConfig.llama_cpp_binary;
    }
    options.privileged_overrides = true;
  }
  applyPresetOptions(options);
}

chatForm?.addEventListener("submit", sendChatMessage);
chatResetBtn?.addEventListener("click", resetChatSession);
chatDownloadBtn?.addEventListener("click", downloadSelectedModel);
chatHfSearchBtn?.addEventListener("click", searchHuggingFaceRepos);
chatHfSearch?.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    event.preventDefault();
    searchHuggingFaceRepos();
  }
});
chatModel?.addEventListener("change", async () => {
  try {
    await selectModel(chatModel.value || null);
    if (chatModel.value && chatBackend.querySelector('option[value="llama_cpp_rl"]')) {
      chatBackend.value = "llama_cpp_rl";
    }
  } catch (error) {
    renderChatMessage("system", error.message || "Could not select model.");
  }
});
chatRunContinuousBtn?.addEventListener("click", () => {
  prefillContinuousFromChat();
  window.scrollTo({ top: 0, behavior: "smooth" });
});
chatBackend?.addEventListener("change", () => {
  if (chatBackend.value === "continuous_learn" || chatBackend.value === "llama_cpp_rl") {
    chatMeta.textContent =
      "Policy update mode: each message applies an RL update and can queue JSONL tasks.";
  } else if (chatBackend.value === "llama_cpp") {
    chatMeta.textContent = "Completion mode: generate text from the selected model.";
  } else {
    fetchChatConfig();
  }
});

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    chatInput?.blur();
  }
});
