"use strict";

const $ = (id) => document.getElementById(id);
const el = (tag, className, text) => {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
};

const STEPS = ["upload", "decode", "layout", "recognize", "done"];
const ZOOMS = [0.5, 0.75, 1, 1.25, 1.5, 2, 3];
const LIMIT_BYTES = 25 * 1024 * 1024;

const state = {
  file: null,
  mode: "auto",
  controller: null,
  runStart: 0,
  kind: "image",
  pageTotal: 0,
  pages: [],
  totalBoxes: 0,
  recognized: 0,
  events: 0,
  final: null,
  active: null,
  zoom: 2,
};

/* ---------- Theme ---------- */
const storedTheme = localStorage.getItem("wormocr-theme");
document.documentElement.dataset.theme =
  storedTheme || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
$("theme-toggle").addEventListener("click", () => {
  const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  localStorage.setItem("wormocr-theme", next);
});

/* ---------- Model ---------- */
fetch("/api/health")
  .then((response) => response.json())
  .then(({ model }) => {
    $("model-dot").className = "dot ok";
    $("model-label").textContent =
      `${model.name} · ${(model.parameters / 1e6).toFixed(2)}M params · ${String(model.device).toUpperCase()}`;
    $("model-label").title = `${model.architecture} · maturity: ${model.maturity}`;
  })
  .catch(() => {
    $("model-dot").className = "dot bad";
    $("model-label").textContent = "Server offline";
  });

/* ---------- Toast ---------- */
let toastTimer;
function toast(message, kind = "") {
  const node = $("toast");
  node.textContent = message;
  node.className = `toast show ${kind}`;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (node.className = `toast ${kind}`), 2600);
}

/* ---------- Files ---------- */
function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function pickFile(file) {
  if (!file) return;
  const isPdf = file.type === "application/pdf" || /\.pdf$/i.test(file.name);
  const isImage = file.type.startsWith("image/") || /\.(png|jpe?g|bmp|tiff?|webp|gif)$/i.test(file.name);
  if (!isPdf && !isImage) return toast("Choose an image or a PDF file.", "error");
  if (file.size > LIMIT_BYTES) return toast("Files up to 25 MB are accepted.", "error");
  state.file = file;
  $("file").hidden = false;
  $("file-name").textContent = file.name;
  $("file-name").title = file.name;
  $("file-meta").textContent = `${formatBytes(file.size)} · ${isPdf ? "PDF" : (file.type || "image").replace("image/", "").toUpperCase()}`;
  $("rerun-button").disabled = false;
  run();
}

$("open-button").addEventListener("click", () => $("file-input").click());
$("file-input").addEventListener("change", (event) => {
  pickFile(event.target.files[0]);
  event.target.value = "";
});
$("rerun-button").addEventListener("click", () => run());

let dragDepth = 0;
addEventListener("dragenter", (event) => {
  if (!event.dataTransfer?.types.includes("Files")) return;
  event.preventDefault();
  dragDepth += 1;
  $("veil").hidden = false;
});
addEventListener("dragover", (event) => event.preventDefault());
addEventListener("dragleave", () => {
  dragDepth = Math.max(0, dragDepth - 1);
  if (!dragDepth) $("veil").hidden = true;
});
addEventListener("drop", (event) => {
  event.preventDefault();
  dragDepth = 0;
  $("veil").hidden = true;
  pickFile(event.dataTransfer?.files[0]);
});
addEventListener("paste", (event) => {
  const file = [...(event.clipboardData?.files || [])].find((item) => item.type.startsWith("image/"));
  if (file) pickFile(new File([file], "pasted-image.png", { type: file.type }));
});

/* ---------- Mode, tabs, toggles ---------- */
document.querySelectorAll(".segmented button").forEach((button) =>
  button.addEventListener("click", () => {
    document.querySelectorAll(".segmented button").forEach((b) => b.setAttribute("aria-checked", String(b === button)));
    state.mode = button.dataset.mode;
    if (state.file) run();
  })
);

document.querySelectorAll(".tabs button").forEach((button) =>
  button.addEventListener("click", () => {
    document.querySelectorAll(".tabs button").forEach((b) => b.setAttribute("aria-selected", String(b === button)));
    document.querySelectorAll("[data-panel]").forEach((panel) => (panel.hidden = panel.dataset.panel !== button.dataset.tab));
  })
);
const currentTab = () => document.querySelector('.tabs button[aria-selected="true"]').dataset.tab;

$("toggle-boxes").addEventListener("change", (event) =>
  document.querySelectorAll(".boxes").forEach((node) => node.classList.toggle("off", !event.target.checked))
);
$("show-lines").addEventListener("change", (event) => $("log").classList.toggle("hide-lines", !event.target.checked));

function applyZoom() {
  const zoom = ZOOMS[state.zoom];
  document.querySelectorAll(".page").forEach((page) => (page.style.width = `${zoom * 100}%`));
  $("pages").style.alignItems = zoom > 1 ? "flex-start" : "center";
  $("zoom-label").textContent = zoom === 1 ? "Fit" : `${Math.round(zoom * 100)}%`;
}
$("zoom-in").addEventListener("click", () => {
  state.zoom = Math.min(ZOOMS.length - 1, state.zoom + 1);
  applyZoom();
});
$("zoom-out").addEventListener("click", () => {
  state.zoom = Math.max(0, state.zoom - 1);
  applyZoom();
});

/* ---------- Status bar ---------- */
function setStep(name, status) {
  const node = document.querySelector(`.steps li[data-step="${name}"]`);
  node.classList.remove("active", "done", "error");
  if (status) node.classList.add(status);
}
function advanceTo(name) {
  const index = STEPS.indexOf(name);
  STEPS.forEach((step, i) => {
    if (i < index) setStep(step, "done");
    else if (i === index) setStep(step, step === "done" ? "done" : "active");
    else setStep(step, null);
  });
}
function failStep() {
  const active = document.querySelector(".steps li.active");
  if (active) setStep(active.dataset.step, "error");
}
function setProgress(fraction, complete = false) {
  const bar = $("progress-bar");
  bar.style.width = `${Math.max(0, Math.min(1, fraction)) * 100}%`;
  bar.classList.toggle("complete", complete);
}

/* ---------- Activity log ---------- */
function log(tag, message, { ms, level, kind } = {}) {
  const list = $("log");
  const body = list.parentElement;
  const stick = body.scrollTop + body.clientHeight >= body.scrollHeight - 30;
  const row = el("li", [level, kind].filter(Boolean).join(" "));
  row.append(
    el("span", "ts", `${((performance.now() - state.runStart) / 1000).toFixed(2)}s`),
    el("span", "tag", tag),
  );
  const msg = el("span", "msg", message);
  if (ms !== undefined) msg.append(el("span", "ms", `  ${ms < 1000 ? `${ms.toFixed(0)} ms` : `${(ms / 1000).toFixed(2)} s`}`));
  row.append(msg);
  list.append(row);
  state.events += 1;
  $("event-count").textContent = `${state.events}`;
  if (stick) body.scrollTop = body.scrollHeight;
}

/* ---------- Run ---------- */
function reset() {
  Object.assign(state, { pages: [], pageTotal: 0, totalBoxes: 0, recognized: 0, events: 0, final: null, active: null });
  advanceTo(null);
  STEPS.forEach((step) => setStep(step, null));
  setProgress(0);
  $("recognize-count").textContent = "";
  $("timings").textContent = "";
  $("log").replaceChildren();
  $("event-count").textContent = "";
  $("pages").replaceChildren();
  $("empty").hidden = true;
  $("page-count").textContent = "";
  $("text-view").replaceChildren(el("p", "placeholder", "Extracted text appears here, line by line."));
  $("lines-body").replaceChildren();
  $("line-count").textContent = "0";
  $("json-view").textContent = "{}";
  ["stat-lines", "stat-confidence", "stat-time", "stat-pages"].forEach((id) => ($(id).textContent = "–"));
  $("doc-body").scrollTop = 0;
}

async function run() {
  if (!state.file) return;
  state.controller?.abort();
  const controller = new AbortController();
  state.controller = controller;
  reset();
  state.runStart = performance.now();
  $("live-dot").className = "live on";
  advanceTo("upload");
  setProgress(0.03);
  log("client", `Uploading ${state.file.name} (${formatBytes(state.file.size)}), mode: ${state.mode}`);

  try {
    const url = `/api/ocr/stream?mode=${encodeURIComponent(state.mode)}&filename=${encodeURIComponent(state.file.name)}`;
    const response = await fetch(url, { method: "POST", body: state.file, signal: controller.signal });
    if (!response.ok) {
      const payload = await response.json().catch(() => ({}));
      throw new Error(payload.error || `Server returned ${response.status}`);
    }
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let newline;
      while ((newline = buffer.indexOf("\n")) >= 0) {
        const line = buffer.slice(0, newline).trim();
        buffer = buffer.slice(newline + 1);
        if (line && state.controller === controller) handle(JSON.parse(line));
      }
    }
    if (state.controller === controller && !state.final && !document.querySelector(".steps li.error")) {
      throw new Error("The connection closed before recognition finished.");
    }
  } catch (error) {
    if (error.name === "AbortError") return;
    failStep();
    log("error", error.message, { level: "error" });
    toast(error.message, "error");
  } finally {
    if (state.controller === controller) $("live-dot").className = "live";
  }
}

/* ---------- Stream events ---------- */
function handle(event) {
  switch (event.type) {
    case "start":
      advanceTo("decode");
      setProgress(0.06);
      break;
    case "log":
      log(event.stage, event.message, { ms: event.ms, level: event.level });
      break;
    case "document":
      state.kind = event.kind;
      state.pageTotal = event.pages;
      $("stat-pages").textContent = String(event.pages);
      $("page-count").textContent = `${event.pages} page${event.pages === 1 ? "" : "s"}`;
      break;
    case "page":
      addPage(event);
      advanceTo("layout");
      break;
    case "layout":
      addBoxes(event);
      advanceTo("recognize");
      break;
    case "line":
      addLine(event);
      break;
    case "done":
      finish(event);
      break;
    case "error":
      failStep();
      log("error", event.message, { level: "error" });
      toast(event.message, "error");
      if (!state.pages.length) $("empty").hidden = false;
      break;
    default:
      break;
  }
}

function addPage(event) {
  const page = { width: event.width, height: event.height, boxes: [], lines: [], layout: "" };
  state.pages[event.page] = page;
  const wrapper = el("div", "page");
  wrapper.id = `page-${event.page}`;
  wrapper.style.width = `${ZOOMS[state.zoom] * 100}%`;
  if (state.pageTotal > 1) wrapper.append(el("div", "page-label", `Page ${event.page + 1} of ${state.pageTotal}`));
  const sheet = el("div", "sheet");
  const img = el("img");
  img.src = event.preview;
  img.alt = `Page ${event.page + 1}`;
  img.style.aspectRatio = `${event.width} / ${event.height}`;
  const boxes = el("div", `boxes ${$("toggle-boxes").checked ? "" : "off"}`);
  boxes.id = `boxes-${event.page}`;
  sheet.append(img, boxes);
  wrapper.append(sheet);
  $("pages").append(wrapper);
}

function addBoxes(event) {
  const page = state.pages[event.page];
  page.layout = event.layout;
  page.boxes = event.boxes;
  state.totalBoxes += event.boxes.length;
  if (event.layout === "line") return;
  const layer = $(`boxes-${event.page}`);
  layer.replaceChildren(
    ...event.boxes.map((b, order) => {
      const box = el("div", "box");
      box.dataset.key = `${event.page}:${order}`;
      box.style.left = `${(b.x / page.width) * 100}%`;
      box.style.top = `${(b.y / page.height) * 100}%`;
      box.style.width = `${(b.width / page.width) * 100}%`;
      box.style.height = `${(b.height / page.height) * 100}%`;
      box.addEventListener("click", () => select(box.dataset.key, "box"));
      return box;
    })
  );
}

function tone(confidence) {
  return confidence >= 0.85 ? "" : confidence >= 0.6 ? "mid" : "low";
}
const pct = (value) => `${(value * 100).toFixed(1)}%`;

function addLine(line) {
  const page = state.pages[line.page];
  page.lines[line.order] = line;
  state.recognized += 1;
  const key = `${line.page}:${line.order}`;
  const level = tone(line.confidence);

  const box = document.querySelector(`.box[data-key="${key}"]`);
  if (box) {
    box.classList.add("done");
    if (level) box.classList.add(level);
    box.title = `${line.text} · ${pct(line.confidence)}`;
  }

  const view = $("text-view");
  if (state.recognized === 1) view.replaceChildren();
  if (state.pageTotal > 1 && !$(`text-page-${line.page}`)) {
    const head = el("div", "text-page-head", `Page ${line.page + 1}`);
    head.id = `text-page-${line.page}`;
    view.append(head);
  }
  const row = el("div", `tl ${level}`);
  row.dataset.key = key;
  row.title = `Confidence ${pct(line.confidence)}`;
  row.append(el("span", "n", String(line.order + 1)), el("span", "t", line.text || " "));
  row.addEventListener("click", () => select(key, "text"));
  view.append(row);

  const tr = el("tr");
  tr.dataset.key = key;
  tr.append(
    el("td", "n", String(line.page + 1)),
    el("td", "n", String(line.order + 1)),
    el("td", "t", line.text || "∅"),
    el("td", `n ${level}`, pct(line.confidence)),
    el("td", "n", String(line.frames)),
  );
  tr.addEventListener("click", () => select(key, "table"));
  $("lines-body").append(tr);

  $("line-count").textContent = String(state.recognized);
  $("stat-lines").textContent = String(state.recognized);
  $("recognize-count").textContent = state.totalBoxes ? `${state.recognized}/${state.totalBoxes}` : "";
  const pagesDone = Math.max(0, line.page) / Math.max(1, state.pageTotal);
  const withinPage = page.boxes.length ? (line.order + 1) / page.boxes.length / Math.max(1, state.pageTotal) : 0;
  setProgress(0.08 + 0.9 * (pagesDone + withinPage));
  const where = state.pageTotal > 1 ? `p${line.page + 1} #${line.order + 1}` : `#${line.order + 1}`;
  log("line", `${where} ${pct(line.confidence)}  "${line.text}"`, { kind: "line" });
}

function finish(event) {
  state.final = event;
  advanceTo("done");
  setProgress(1, true);
  $("recognize-count").textContent = "";
  $("stat-lines").textContent = String(event.line_count);
  $("stat-confidence").textContent = event.mean_confidence == null ? "–" : pct(event.mean_confidence);
  $("stat-time").textContent = `${(event.elapsed_ms / 1000).toFixed(2)} s`;
  const t = event.timings;
  const fmt = (ms) => (ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(2)} s`);
  $("timings").textContent =
    `decode ${fmt(t.decode)} · layout ${fmt(t.layout)} · recognize ${fmt(t.recognize)} · total ${fmt(event.elapsed_ms)}`;
  if (!state.recognized) $("text-view").replaceChildren(el("p", "placeholder", "No text was found."));
  $("json-view").textContent = JSON.stringify(result(), null, 2);
  const lastLog = $("log").lastElementChild;
  if (lastLog) lastLog.classList.add("done");
  if (event.warnings?.length) toast(event.warnings[0]);
}

/* ---------- Selection ---------- */
function select(key, source) {
  state.active = key;
  document.querySelectorAll("[data-key].active").forEach((node) => node.classList.remove("active"));
  document.querySelectorAll(`[data-key="${key}"]`).forEach((node) => node.classList.add("active"));
  const reveal = (node) => node?.scrollIntoView({ block: "nearest", behavior: "smooth" });
  if (source !== "box") reveal(document.querySelector(`.box[data-key="${key}"]`) || $(`page-${key.split(":")[0]}`));
  if (source !== "text") reveal(document.querySelector(`.tl[data-key="${key}"]`));
  if (source !== "table") reveal(document.querySelector(`tr[data-key="${key}"]`));
}

/* ---------- Export ---------- */
function plainText() {
  return state.pages.map((page) => page.lines.filter(Boolean).map((line) => line.text).join("\n")).join("\n\n");
}

function result() {
  return {
    filename: state.file?.name,
    mode: state.mode,
    kind: state.kind,
    text: plainText(),
    line_count: state.final?.line_count ?? state.recognized,
    mean_confidence: state.final?.mean_confidence ?? null,
    elapsed_ms: state.final?.elapsed_ms ?? null,
    timings_ms: state.final?.timings ?? null,
    warnings: state.final?.warnings ?? [],
    pages: state.pages.map((page, index) => ({
      page: index + 1,
      width: page.width,
      height: page.height,
      layout: page.layout,
      lines: page.lines.filter(Boolean).map(({ order, text, confidence, bbox, frames }) => ({
        order: order + 1,
        text,
        confidence: Number(confidence.toFixed(4)),
        bbox,
        ctc_steps: frames,
      })),
    })),
  };
}

$("copy-button").addEventListener("click", async () => {
  const asJson = currentTab() === "json";
  const content = asJson ? JSON.stringify(result(), null, 2) : plainText();
  if (!content) return toast("Nothing to copy yet.");
  await navigator.clipboard.writeText(content);
  toast(asJson ? "JSON copied." : "Text copied.");
});

$("download-button").addEventListener("click", () => {
  if (!state.recognized) return toast("Nothing to download yet.");
  const asJson = currentTab() === "json";
  const base = (state.file?.name || "wormocr").replace(/\.[^.]+$/, "");
  const blob = asJson
    ? new Blob([JSON.stringify(result(), null, 2)], { type: "application/json" })
    : new Blob([plainText()], { type: "text/plain" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = `${base}.${asJson ? "json" : "txt"}`;
  link.click();
  setTimeout(() => URL.revokeObjectURL(link.href), 1000);
});

/* ---------- Sample ---------- */
$("sample-button").addEventListener("click", async () => {
  const canvas = document.createElement("canvas");
  canvas.width = 1240;
  canvas.height = 900;
  const g = canvas.getContext("2d");
  g.fillStyle = "#fff";
  g.fillRect(0, 0, canvas.width, canvas.height);
  g.fillStyle = "#111";
  [
    ["bold 44px Arial", "INVOICE 2026-1042", 90],
    ["26px Arial", "Acme Supplies Ltd, 42 Market Street, Springfield", 150],
    ["26px Arial", "Phone: (555) 013-2244   Email: billing@acme.com", 196],
    ["bold 28px Arial", "Bill to: Northwind Traders", 282],
    ["26px Arial", "Date: 22 Oct 2026   Due: 21 Nov 2026", 330],
    ["26px Arial", "Printer paper A4, 12 boxes at $4.25 each", 420],
    ["26px Arial", "Toner cartridge, 2 units at $38.90 each", 468],
    ["26px Arial", "Delivery and handling: $12.00", 516],
    ["26px Arial", "Subtotal: $140.80", 606],
    ["26px Arial", "Tax (8%): $11.26", 654],
    ["bold 30px Arial", "Total due: $152.06", 712],
    ["24px Arial", "Thank you for your business!", 806],
  ].forEach(([font, text, y]) => {
    g.font = font;
    g.fillText(text, 90, y);
  });
  const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/png"));
  pickFile(new File([blob], "sample-invoice.png", { type: "image/png" }));
});

if (location.hash === "#sample") $("sample-button").click();
