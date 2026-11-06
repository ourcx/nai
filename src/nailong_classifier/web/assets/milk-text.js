"use strict";

const form = document.querySelector("#editor-form");
const textField = document.querySelector("#sticker-text");
const preview = document.querySelector("#sticker-preview");
const placeholder = document.querySelector("#preview-placeholder");
const stage = document.querySelector("#preview-stage");
const status = document.querySelector("#status");
const copyButton = document.querySelector("#copy-image");
const downloadButton = document.querySelector("#download-image");
let objectUrl = null;
let currentBlob = null;
let timer = null;
let pending = null;
let revision = 0;
let scheduledOptions = null;

function setStatus(message, error = false) {
  status.textContent = message;
  status.classList.toggle("is-error", error);
}

function options() {
  const state = Object.fromEntries(new FormData(form));
  for (const key of ["font_size", "stroke", "tilt", "amplitude", "duration_ms"]) {
    // 留空交给服务端均匀随机；显式的 0 仍保留其自动适配或零幅度含义。
    const input = form.elements.namedItem(key);
    if (input.validity.badInput) {
      state[key] = "invalid";
    } else if (state[key] === "") {
      delete state[key];
    } else {
      state[key] = Number(state[key]);
    }
  }
  return state;
}

function disableExport() {
  currentBlob = null;
  copyButton.disabled = true;
  downloadButton.disabled = true;
}

async function render() {
  const currentRevision = ++revision;
  pending?.abort();
  disableExport();
  const state = options();
  stage.classList.toggle("is-cream", state.background === "cream");
  if (!state.text.trim()) {
    preview.hidden = true;
    placeholder.hidden = false;
    placeholder.textContent = "写一句话，就能变成奶字。";
    document.querySelector("#output-size").textContent = "等你写一句";
    stage.classList.remove("is-loading");
    setStatus("");
    return;
  }
  const controller = new AbortController();
  pending = controller;
  stage.classList.add("is-loading");
  setStatus("正在制作…");
  try {
    const response = await fetch("../api/milk-text", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(state),
      signal: controller.signal,
    });
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || "这次没做成功，再试一下。");
    }
    const blob = await response.blob();
    if (currentRevision !== revision) return;
    const nextUrl = URL.createObjectURL(blob);
    // Decode first: never let an old response replace the latest text, and keep exports current.
    const image = new Image();
    image.src = nextUrl;
    try {
      await image.decode();
    } catch (error) {
      URL.revokeObjectURL(nextUrl);
      throw error;
    }
    if (currentRevision !== revision) {
      URL.revokeObjectURL(nextUrl);
      return;
    }
    if (objectUrl) URL.revokeObjectURL(objectUrl);
    objectUrl = nextUrl;
    currentBlob = blob;
    preview.src = objectUrl;
    preview.alt = `奶系文字贴纸：${state.text}`;
    preview.hidden = false;
    placeholder.hidden = true;
    document.querySelector("#output-size").textContent = `${image.naturalWidth} × ${image.naturalHeight} px`;
    const animated = blob.type === "image/gif";
    document.querySelector("#export-format").textContent = animated
      ? "GIF 动图 · 下载后发到群里" : "PNG · 放进聊天，刚刚好";
    downloadButton.disabled = false;
    copyButton.disabled = animated;
    copyButton.title = animated ? "动图请下载 GIF 后发送" : "";
    setStatus("做好了，可以直接发到群里。");
  } catch (error) {
    if (error.name === "AbortError" || currentRevision !== revision) return;
    preview.hidden = true;
    placeholder.hidden = false;
    placeholder.textContent = "改一下文字，再试试。";
    document.querySelector("#output-size").textContent = "暂未生成";
    setStatus(error.message || "图片生成失败，请重试。", true);
  } finally {
    if (currentRevision === revision) stage.classList.remove("is-loading");
  }
}

function scheduleRender() {
  // 文本框失焦会再触发 change；内容未变时保留成图，避免第一次点击下载被禁用按钮吃掉。
  const nextOptions = JSON.stringify(options());
  if (nextOptions === scheduledOptions) return;
  scheduledOptions = nextOptions;
  clearTimeout(timer);
  // Invalidate on input, before debounce, so a pending old response cannot enable stale exports.
  revision++;
  pending?.abort();
  disableExport();
  const hasText = Boolean(textField.value.trim());
  stage.classList.toggle("is-loading", hasText);
  setStatus(hasText ? "正在制作…" : "");
  document.querySelector("#char-count").textContent = `${Array.from(textField.value).length} / 60`;
  timer = setTimeout(render, 180);
}

form.addEventListener("submit", event => event.preventDefault());
form.addEventListener("input", scheduleRender);
form.addEventListener("change", scheduleRender);
document.querySelectorAll('input[name="background"]').forEach(input => {
  input.addEventListener("change", scheduleRender);
});
document.querySelectorAll("[data-preset]").forEach(button => {
  button.addEventListener("click", () => {
    textField.value = button.dataset.preset;
    scheduleRender();
  });
});
document.querySelector("#randomize").addEventListener("click", () => {
  scheduledOptions = null;
  scheduleRender();
});
downloadButton.addEventListener("click", () => {
  if (!currentBlob || !objectUrl) return;
  const link = document.createElement("a");
  link.href = objectUrl;
  const format = currentBlob.type === "image/gif" ? "GIF" : "PNG";
  link.download = `奶峰表情包.${format.toLowerCase()}`;
  document.body.appendChild(link);
  link.click();
  link.remove();
  setStatus(`已下载 ${format}，发到群里试试吧。`);
});
copyButton.addEventListener("click", async () => {
  if (!currentBlob) return;
  if (!window.isSecureContext || !navigator.clipboard?.write || !window.ClipboardItem) {
    setStatus("这个浏览器暂不能复制图片，请点击「下载表情包」。");
    return;
  }
  const blob = currentBlob;
  try {
    await navigator.clipboard.write([new ClipboardItem({ "image/png": blob })]);
    setStatus("图片已复制，去聊天里粘贴吧。");
  } catch {
    setStatus("没有拿到剪贴板权限，请下载图片后发送。");
  }
});
window.addEventListener("pagehide", () => {
  pending?.abort();
  if (objectUrl) URL.revokeObjectURL(objectUrl);
});
if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
  document.querySelector("#motion").value = "static";
}
scheduleRender();
