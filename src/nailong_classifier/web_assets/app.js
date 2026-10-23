const gallery = document.querySelector("#gallery");
const emptyState = document.querySelector("#empty-state");
const loadMoreButton = document.querySelector("#load-more");
const sortSelect = document.querySelector("#sort-select");
const resultCount = document.querySelector("#result-count");
const collectionTitle = document.querySelector("#collection-title");
const filterButtons = [...document.querySelectorAll(".stat-card")];

const dialog = document.querySelector("#image-dialog");
const dialogClose = document.querySelector("#dialog-close");
const dialogImage = document.querySelector("#dialog-image");
const dialogLoader = document.querySelector("#dialog-loader");
const dialogLabel = document.querySelector("#dialog-label");
const dialogTitle = document.querySelector("#dialog-title");
const dialogDate = document.querySelector("#dialog-date");
const dialogSeen = document.querySelector("#dialog-seen");
const dialogDiscovererRow = document.querySelector("#dialog-discoverer-row");
const dialogDiscoverer = document.querySelector("#dialog-discoverer");
const dialogSource = document.querySelector("#dialog-source");
const dialogOriginal = document.querySelector("#dialog-original");

const labelTitles = {
  "": "全部收藏",
  nailong: "奶龙",
  naiwa: "奶蛙",
};

const state = {
  label: "",
  sort: "newest",
  page: 1,
  pageSize: 36,
  hasMore: false,
  loading: false,
  controller: null,
};

const dateFormatter = new Intl.DateTimeFormat("zh-CN", {
  year: "numeric",
  month: "short",
  day: "numeric",
  hour: "2-digit",
  minute: "2-digit",
});

function formatDate(value) {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? "时间未知" : dateFormatter.format(date);
}

function makeElement(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined) element.textContent = text;
  return element;
}

function showSkeletons() {
  gallery.replaceChildren();
  for (let index = 0; index < 8; index += 1) {
    const skeleton = makeElement("div", "skeleton");
    skeleton.setAttribute("aria-hidden", "true");
    gallery.append(skeleton);
  }
}

function makeCard(asset) {
  const card = makeElement("button", "image-card");
  card.type = "button";
  card.setAttribute("aria-label", `查看${asset.label_name}图片 #${String(asset.id).padStart(4, "0")}`);

  const frame = makeElement("div", "image-frame");
  const image = document.createElement("img");
  image.src = asset.thumbnail_url;
  image.alt = `${asset.label_name} #${String(asset.id).padStart(4, "0")}`;
  image.loading = "lazy";
  image.decoding = "async";
  image.fetchPriority = "low";
  image.addEventListener("error", () => image.classList.add("is-broken"));
  frame.append(image);

  const meta = makeElement("div", "card-meta");
  const topLine = makeElement("div", "card-topline");
  const label = makeElement("span", "label-pill", asset.label_name);
  label.dataset.label = asset.label;
  const number = makeElement("span", "asset-number", `#${String(asset.id).padStart(4, "0")}`);
  topLine.append(label, number);

  const bottomLine = makeElement("div", "card-bottomline");
  bottomLine.append(
    makeElement("span", "", formatDate(asset.first_seen_at)),
    makeElement("span", "", `出现 ${asset.seen_count} 次`),
  );
  meta.append(topLine, bottomLine);
  card.append(frame, meta);
  card.addEventListener("click", () => openDialog(asset));
  return card;
}

function openDialog(asset) {
  dialogLabel.textContent = asset.label_name;
  dialogLabel.dataset.label = asset.label;
  dialogTitle.textContent = `图鉴 #${String(asset.id).padStart(4, "0")}`;
  dialogDate.textContent = formatDate(asset.first_seen_at);
  dialogSeen.textContent = `${asset.seen_count} 次`;
  dialogDiscovererRow.hidden = !asset.discoverer_name;
  dialogDiscoverer.textContent = asset.discoverer_name || "";
  dialogSource.textContent = asset.corrected
    ? "人工确认"
    : asset.origin === "seed"
      ? "初始图库"
      : `模型识别 · ${Math.round(asset.confidence * 100)}%`;
  dialogOriginal.href = asset.original_url;
  dialogImage.alt = `${asset.label_name} #${String(asset.id).padStart(4, "0")} 原图`;
  dialogLoader.textContent = "原图加载中…";
  dialogLoader.hidden = false;
  dialogImage.removeAttribute("src");
  dialog.showModal();
  dialogImage.src = asset.original_url;
}

function closeDialog() {
  dialog.close();
  dialogImage.removeAttribute("src");
  dialogLoader.hidden = false;
}

dialogImage.addEventListener("load", () => {
  dialogLoader.hidden = true;
});

dialogImage.addEventListener("error", () => {
  dialogLoader.textContent = "原图暂时无法加载";
});

dialogClose.addEventListener("click", closeDialog);
dialog.addEventListener("click", (event) => {
  if (event.target === dialog) closeDialog();
});
dialog.addEventListener("close", () => dialogImage.removeAttribute("src"));

async function loadStats() {
  const response = await fetch("api/stats?v=2");
  if (!response.ok) throw new Error("统计加载失败");
  const stats = await response.json();
  for (const name of ["total", "nailong", "naiwa"]) {
    const element = document.querySelector(`[data-stat="${name}"]`);
    element.textContent = new Intl.NumberFormat("zh-CN").format(stats[name]);
  }
}

async function loadAssets({ append = false } = {}) {
  if (state.controller) state.controller.abort();
  const controller = new AbortController();
  state.controller = controller;
  state.loading = true;
  loadMoreButton.disabled = true;
  if (!append) showSkeletons();

  const parameters = new URLSearchParams({
    page: String(state.page),
    page_size: String(state.pageSize),
    sort: state.sort,
  });
  if (state.label) parameters.set("label", state.label);

  try {
    parameters.set("v", "2");
    const response = await fetch(`api/assets?${parameters}`, {
      signal: controller.signal,
    });
    if (!response.ok) throw new Error("图片加载失败");
    const payload = await response.json();
    if (!append) gallery.replaceChildren();
    const fragment = document.createDocumentFragment();
    for (const asset of payload.items) fragment.append(makeCard(asset));
    gallery.append(fragment);
    state.hasMore = payload.has_more;
    resultCount.textContent = new Intl.NumberFormat("zh-CN").format(payload.total);
    emptyState.hidden = payload.total !== 0;
    loadMoreButton.hidden = !payload.has_more;
  } catch (error) {
    if (error.name === "AbortError") return;
    if (!append) gallery.replaceChildren();
    emptyState.hidden = false;
    emptyState.querySelector("h3").textContent = "图片没有加载出来";
    emptyState.querySelector("p").textContent = "稍后刷新页面再试一次。";
  } finally {
    if (state.controller !== controller) return;
    state.loading = false;
    loadMoreButton.disabled = false;
    loadMoreButton.textContent = "再看一些";
  }
}

function selectLabel(label) {
  state.label = label;
  state.page = 1;
  collectionTitle.textContent = labelTitles[label];
  for (const button of filterButtons) {
    const active = button.dataset.label === label;
    button.classList.toggle("is-active", active);
    button.setAttribute("aria-pressed", String(active));
  }
  loadAssets();
}

for (const button of filterButtons) {
  button.addEventListener("click", () => selectLabel(button.dataset.label));
}

sortSelect.addEventListener("change", () => {
  state.sort = sortSelect.value;
  state.page = 1;
  loadAssets();
});

loadMoreButton.addEventListener("click", () => {
  if (!state.hasMore) return;
  state.page += 1;
  loadMoreButton.textContent = "正在加载…";
  loadAssets({ append: true });
});

Promise.allSettled([loadStats(), loadAssets()]);
