// Copyright The KiLoRA Authors
// SPDX-License-Identifier: Apache-2.0
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const LOCALES = ["en", "ar", "de", "es", "fa", "fr", "he", "it", "ja", "ko", "pt-BR", "ru", "tr", "zh", "zh-TW"];
export const MACHINE_MADE = LOCALES.filter((l) => l !== "en");
let STRINGS = {};
let LANG = "en";
const RTL = new Set(["ar", "fa", "he"]);

function pickLocale(raw) {
  const v = String(raw || "en");
  if (LOCALES.includes(v)) return v;
  const low = v.toLowerCase();
  if (low.startsWith("zh")) return /tw|hk|mo|hant/.test(low) ? "zh-TW" : "zh";
  if (low.startsWith("pt")) return "pt-BR";
  const base = low.split(/[-_]/)[0];
  return LOCALES.includes(base) ? base : "en";
}

async function loadStrings() {
  const setting = app.extensionManager?.setting?.get?.("Comfy.Locale") ?? app.ui?.settings?.getSettingValue?.("Comfy.Locale");
  const lang = pickLocale(setting || navigator.language);
  LANG = lang;
  const read = async (l) => (await fetch(new URL(`./i18n/${l}.json`, import.meta.url))).json();
  try {
    STRINGS = await read("en");
    if (lang !== "en") STRINGS = { ...STRINGS, ...(await read(lang)) };
  } catch {
    STRINGS = {};
  }
}

function t(key, values = {}) {
  const s = STRINGS[key] ?? key;
  return s.replace(/\{(\w+)\}/g, (_, k) => (values[k] ?? ""));
}

const PREFIX = "KiLoRA";
const COLORS = {
  title: "#2a2722",
  body: "#1f1d19",
  accent: "#e8461c",
  off: "#8a8577",
  paused: "#e0a33a",
  liveEdge: "#e8461c",
  text: "#ede7d9",
};

const LIVE_NODES = new Set([
  "KiLoRALiveOptions", "KiLoRALivePacks", "KiLoRALiveSave",
  "KiLoRALiveRecord", "KiLoRALiveWrite",
]);

const PATTERNS = {
  undo: [/(\d+)\s+undo entr(?:y|ies)/i, /undo(?:\s+depth)?\s*[:=]\s*(\d+)/i, /(\d+)\s+undo steps?/i],
  writes: [/(\d+)\s+writes?\b(?:\s+since\s+reset)?/i, /writes?(?:\s+since\s+reset)?\s*[:=]\s*(\d+)/i],
};

function firstNumber(list, text) {
  for (const re of list) {
    const m = re.exec(text);
    if (m) return Number(m[1]);
  }
  return undefined;
}

function parseReport(text) {
  const out = {};
  const undo = firstNumber(PATTERNS.undo, text);
  if (undo !== undefined) out.undo = undo;
  const writes = firstNumber(PATTERNS.writes, text);
  if (writes !== undefined) out.writes = writes;
  if (/\breset\b[^.]*\b(?:empty|shipped|clean|start)/i.test(text)) out.writes = 0;
  const learning = /learning\s*(?:is\s*)?(on|off)\b/i.exec(text);
  if (learning) out.learning = learning[1].toLowerCase() === "on";
  if (/\bpaused\b/i.test(text)) out.learning = false;
  return out;
}

function learningFromWidget(node) {
  const w = node.widgets?.find((x) => x.name === "learning");
  return typeof w?.value === "boolean" ? w.value : undefined;
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  if (ctx.roundRect) ctx.roundRect(x, y, w, h, r);
  else ctx.rect(x, y, w, h);
}

function drawBadge(ctx, node) {
  const label = "KiLoRA";
  ctx.save();
  ctx.font = "bold 10px sans-serif";
  const w = ctx.measureText(label).width + 12;
  const x = node.size[0] - w - 6;
  const y = -LiteGraph.NODE_TITLE_HEIGHT + 7;
  ctx.fillStyle = COLORS.accent;
  roundRect(ctx, x, y, w, 16, 8);
  ctx.fill();
  ctx.fillStyle = "#ffffff";
  ctx.textBaseline = "middle";
  ctx.fillText(label, x + 6, y + 8.5);
  ctx.restore();
}

function drawLiveState(ctx, node) {
  const st = node.kiloraLive || {};
  const learning = learningFromWidget(node) ?? st.learning;
  const known = learning !== undefined || st.undo !== undefined || st.writes !== undefined;
  const n = (v) => (v === undefined ? "–" : String(v));
  const text = !known ? t("web.live.waiting")
    : `${learning === false ? t("web.live.paused") : t("web.live.learning")} · ${t("web.live.writes", { n: n(st.writes) })} · ${t("web.live.undo", { n: n(st.undo) })}`;
  ctx.save();
  ctx.font = "11px sans-serif";
  const w = ctx.measureText(text).width + 34;
  const x = 0, y = node.size[1] + 6, h = 20;
  ctx.fillStyle = "rgba(20, 19, 15, 0.92)";
  roundRect(ctx, x, y, w, h, 10);
  ctx.fill();
  ctx.strokeStyle = learning === false ? COLORS.paused : "rgba(237, 231, 217, 0.18)";
  ctx.lineWidth = 1;
  ctx.stroke();
  const cx = x + 12, cy = y + h / 2;
  if (!known) {
    ctx.strokeStyle = COLORS.off;
    ctx.beginPath(); ctx.arc(cx, cy, 3.5, 0, Math.PI * 2); ctx.stroke();
  } else if (learning === false) {
    ctx.fillStyle = COLORS.paused;
    ctx.fillRect(cx - 4, cy - 4, 3, 8);
    ctx.fillRect(cx + 1, cy - 4, 3, 8);
  } else {
    ctx.fillStyle = COLORS.accent;
    ctx.beginPath(); ctx.arc(cx, cy, 4, 0, Math.PI * 2); ctx.fill();
  }
  ctx.fillStyle = COLORS.text;
  ctx.textBaseline = "middle";
  ctx.fillText(text, x + 24, cy + 0.5);
  ctx.restore();
}

let lastPointer = { x: 120, y: 120 };
document.addEventListener("pointerdown", (e) => { lastPointer = { x: e.clientX, y: e.clientY }; }, true);
let openPop = null;

function closePopover() {
  if (!openPop) return;
  openPop.el.remove();
  document.removeEventListener("keydown", openPop.onKey, true);
  document.removeEventListener("pointerdown", openPop.onDown, true);
  openPop = null;
}

function el(tag, attrs = {}, ...kids) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "text") n.textContent = v;
    else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
    else n.setAttribute(k, v);
  }
  for (const c of kids) if (c) n.append(c);
  return n;
}

const POP_CSS = `
.kilora-pop{position:fixed;z-index:10000;width:300px;max-width:calc(100vw - 24px);background:#1c1b16;color:#ede7d9;border:1px solid #403d33;border-radius:12px;box-shadow:0 18px 40px -16px rgba(0,0,0,.7);font:13px/1.4 system-ui,sans-serif;padding:12px}
.kilora-pop h2{font-size:14px;margin:0 0 8px;display:flex;justify-content:space-between;align-items:center}
.kilora-pop h3{font-size:12px;margin:10px 0 6px;color:#aaa392;font-weight:600}
.kilora-pop ol{list-style:none;margin:0;padding:0;max-height:180px;overflow:auto;display:grid;gap:4px}
.kilora-pop li{padding:6px 8px;border-radius:8px;background:#27261f;font-variant-numeric:tabular-nums}
.kilora-pop li.cur{background:#34322a;box-shadow:inset 3px 0 0 #e8461c}
.kilora-pop .row{display:flex;gap:6px;justify-content:flex-end;margin-top:10px;flex-wrap:wrap}
.kilora-pop button{min-height:32px;padding:0 12px;border-radius:999px;border:1px solid #403d33;background:transparent;color:#ede7d9;cursor:pointer;font:inherit}
.kilora-pop button.primary{background:#ede7d9;color:#1c1b16;border-color:#ede7d9}
.kilora-pop button.danger{background:#b8320c;border-color:#b8320c;color:#fff}
.kilora-pop button:focus-visible{outline:2px solid #9fb2ff;outline-offset:2px}
.kilora-pop .note{color:#aaa392;margin:8px 0 0}
.kilora-pop .editor{border-bottom:1px solid #3a3b33;padding-bottom:8px;margin-bottom:8px}
.kilora-pop .editor p{margin:0;overflow-wrap:anywhere}
.kilora-pop .editor .ver{color:#aaa392}
`;

function editorInfo(node) {
  const nodes = app.graph?._nodes || app.graph?.nodes || [];
  const rec = node.comfyClass === "KiLoRALiveRecord" ? node : nodes.find((n) => n.comfyClass === "KiLoRALiveRecord" || n.type === "KiLoRALiveRecord");
  if (!rec) return { key: "web.pop.editor_none" };
  const v = String(rec.widgets?.find((w) => w.name === "editor")?.value ?? "").trim();
  if (!v) return { key: "web.pop.editor_empty" };
  return { key: "web.pop.editor", name: v.replace(/\.kedit(\.safetensors)?$/i, "") };
}

function showPopover(node) {
  closePopover();
  if (!document.getElementById("kilora-pop-css")) document.head.append(el("style", { id: "kilora-pop-css", text: POP_CSS }));
  const st = node.kiloraLive || {};
  const hist = st.history || [];
  const pop = el("div", { class: "kilora-pop", role: "dialog", "aria-labelledby": "kilora-pop-h",
    dir: RTL.has(LANG) ? "rtl" : "ltr", lang: LANG });
  const body = el("div");
  const list = el("ol", { "aria-label": t("web.pop.history") });
  if (!hist.length) list.append(el("li", { text: t("web.pop.empty") }));
  hist.slice().reverse().forEach((h, i) => list.append(el("li", { class: i === 0 ? "cur" : "",
    text: t("web.pop.entry", { i: h.i, w: h.w ?? "–", u: h.u ?? "–" }) })));
  const note = el("p", { class: "note", "aria-live": "polite" });
  const resetBtn = el("button", { type: "button", class: "danger", text: t("web.pop.reset"), onclick: () => confirmReset() });
  const closeBtn = el("button", { type: "button", text: t("web.pop.close"), onclick: closePopover });
  const ed = editorInfo(node);
  const edBox = el("div", { class: "editor" },
    el("p", { text: t(ed.key, { name: ed.name ?? "" }) }),
    el("p", { class: "ver", text: t("web.pop.editor_version") }));
  body.append(edBox, el("h3", { text: t("web.pop.history") }), list, note, el("div", { class: "row" }, resetBtn, closeBtn));
  pop.append(el("h2", { id: "kilora-pop-h", text: `${t("web.pop.title")} · ${node.title}` }), body);

  function confirmReset() {
    const action = node.widgets?.find((w) => w.name === "action" && (w.options?.values || []).includes("reset"));
    body.replaceChildren(
      el("h3", { text: t("web.pop.reset_title") }),
      el("p", { text: t("web.pop.reset_body") }),
      el("div", { class: "row" },
        el("button", { type: "button", text: t("web.pop.cancel"), onclick: () => showPopover(node) }),
        el("button", { type: "button", class: "danger", text: t("web.pop.confirm"), onclick: () => {
          if (action) {
            action.value = "reset";
            action.callback?.("reset");
            node.setDirtyCanvas?.(true, true);
            showPopover(node);
            openPop?.el.querySelector(".note")?.replaceChildren(t("web.pop.reset_set"));
          } else {
            showPopover(node);
            openPop?.el.querySelector(".note")?.replaceChildren(t("web.pop.reset_where"));
          }
        } })));
    body.querySelector("button")?.focus();
  }

  document.body.append(pop);
  const r = pop.getBoundingClientRect();
  pop.style.left = `${Math.max(12, Math.min(lastPointer.x, innerWidth - r.width - 12))}px`;
  pop.style.top = `${Math.max(12, Math.min(lastPointer.y, innerHeight - r.height - 12))}px`;
  const onKey = (e) => { if (e.key === "Escape") { e.stopPropagation(); closePopover(); } };
  const onDown = (e) => { if (!pop.contains(e.target)) closePopover(); };
  document.addEventListener("keydown", onKey, true);
  setTimeout(() => document.addEventListener("pointerdown", onDown, true), 0);
  openPop = { el: pop, onKey, onDown };
  closeBtn.focus();
}

function isNodeTypeRegistered(typeName) {
  if (typeof LiteGraph === "undefined" || !LiteGraph.registered_node_types) return false;
  return Boolean(LiteGraph.registered_node_types[typeName]);
}

function showNotification(msg) {
  if (app.extensionManager?.toast?.add) {
    app.extensionManager.toast.add({
      severity: "warn",
      summary: "KiLoRA",
      detail: msg,
      life: 5000,
    });
    return;
  }
  if (app.extensionManager?.toast?.addAlert) {
    app.extensionManager.toast.addAlert(msg);
    return;
  }
  if (app.ui?.dialog?.show) {
    app.ui.dialog.show(msg);
    return;
  }
  if (typeof alert === "function") {
    alert(msg);
  }
}

function getSelectedNodes(cv) {
  const raw = cv?.selected_nodes || app.canvas?.selected_nodes;
  if (!raw) return [];
  if (Array.isArray(raw)) return raw.filter(Boolean);
  return Object.values(raw).filter(Boolean);
}

function getWidgetValues(node) {
  let lora_name;
  let strength_model;
  let strength_clip;

  if (Array.isArray(node.widgets)) {
    for (const w of node.widgets) {
      if (w.name === "lora_name") lora_name = w.value;
      else if (w.name === "strength_model") strength_model = w.value;
      else if (w.name === "strength_clip") strength_clip = w.value;
    }
  }

  if (Array.isArray(node.widgets_values)) {
    if (lora_name === undefined) lora_name = node.widgets_values[0];
    if (strength_model === undefined) strength_model = node.widgets_values[1];
    if (strength_clip === undefined && node.widgets_values.length > 2) {
      strength_clip = node.widgets_values[2];
    }
  }

  return { lora_name, strength_model, strength_clip };
}

function applyWidgetValues(newNode, oldNode, values, isModelOnly) {
  const { lora_name, strength_model, strength_clip } = values;

  if (Array.isArray(newNode.widgets) && newNode.widgets.length > 0) {
    for (const w of newNode.widgets) {
      if (w.name === "lora_name" && lora_name !== undefined) {
        w.value = lora_name;
        w.callback?.(lora_name);
      } else if (w.name === "strength_model" && strength_model !== undefined) {
        w.value = strength_model;
        w.callback?.(strength_model);
      } else if (w.name === "strength_clip" && strength_clip !== undefined) {
        w.value = strength_clip;
        w.callback?.(strength_clip);
      }
    }
  } else if (Array.isArray(oldNode.widgets)) {
    newNode.widgets = isModelOnly
      ? [
          { ...(oldNode.widgets[0] || {}), name: "lora_name", value: lora_name },
          { ...(oldNode.widgets[1] || {}), name: "strength_model", value: strength_model },
        ]
      : [
          { ...(oldNode.widgets[0] || {}), name: "lora_name", value: lora_name },
          { ...(oldNode.widgets[1] || {}), name: "strength_model", value: strength_model },
          { ...(oldNode.widgets[2] || oldNode.widgets[1] || {}), name: "strength_clip", value: strength_clip },
        ];
  }

  if (oldNode.widgets_values !== undefined) {
    if (isModelOnly) {
      newNode.widgets_values = [lora_name, strength_model];
    } else {
      newNode.widgets_values = [lora_name, strength_model, strength_clip];
    }
  }
}

function createNodeInstance(type) {
  if (typeof LiteGraph !== "undefined" && typeof LiteGraph.createNode === "function") {
    const node = LiteGraph.createNode(type);
    if (node) return node;
  }

  const isModelOnly = type === "LoraLoaderModelOnly";
  return {
    type,
    inputs: isModelOnly
      ? [{ name: "model", type: "MODEL", link: null }]
      : [{ name: "model", type: "MODEL", link: null }, { name: "clip", type: "CLIP", link: null }],
    outputs: isModelOnly
      ? [{ name: "MODEL", type: "MODEL", links: null }]
      : [{ name: "MODEL", type: "MODEL", links: null }, { name: "CLIP", type: "CLIP", links: null }],
  };
}

function replaceNode(oldNode, reverse, graph, canvas) {
  const oldType = oldNode.type || oldNode.comfyClass;

  let targetType;
  if (!reverse) {
    targetType = "KiLoRALoRALoader";
  } else {
    const clipInput = oldNode.inputs?.find((inp) => inp.name.toLowerCase() === "clip");
    const hasClipInp = Boolean(clipInput && clipInput.link !== null && clipInput.link !== undefined);
    const clipOutput = oldNode.outputs?.find((out) => out.name.toLowerCase() === "clip");
    const hasClipOut = Boolean(clipOutput?.links && clipOutput.links.length > 0);
    const hasClipLink = hasClipInp || hasClipOut;

    if (oldNode.properties?.kiloraConvertedFrom === "LoraLoaderModelOnly") {
      targetType = hasClipLink ? "LoraLoader" : "LoraLoaderModelOnly";
    } else if (oldNode.properties?.kiloraConvertedFrom === "LoraLoader") {
      targetType = "LoraLoader";
    } else {
      targetType = "LoraLoader";
    }
  }

  const newNode = createNodeInstance(targetType);
  newNode.id = oldNode.id;
  newNode.type = targetType;
  if (oldNode.comfyClass !== undefined || newNode.comfyClass !== undefined) {
    newNode.comfyClass = targetType;
  }

  if (oldNode.pos) newNode.pos = [oldNode.pos[0], oldNode.pos[1]];
  if (oldNode.size) newNode.size = [oldNode.size[0], oldNode.size[1]];
  if (oldNode.title !== undefined) newNode.title = oldNode.title;
  if (oldNode.flags) newNode.flags = { ...oldNode.flags };
  if (oldNode.order !== undefined) newNode.order = oldNode.order;
  if (oldNode.mode !== undefined) newNode.mode = oldNode.mode;

  const origProps = oldNode.properties;
  if (!reverse) {
    newNode.properties = origProps ? { ...origProps } : {};
    newNode.properties.kiloraConvertedFrom = oldType;
    if (!origProps) newNode.properties._kiloraNoOrigProps = true;
  } else {
    newNode.properties = origProps ? { ...origProps } : {};
    const hadNoOrig = newNode.properties._kiloraNoOrigProps;
    delete newNode.properties.kiloraConvertedFrom;
    delete newNode.properties._kiloraNoOrigProps;
    if (hadNoOrig && Object.keys(newNode.properties).length === 0) {
      delete newNode.properties;
    }
  }

  const vals = getWidgetValues(oldNode);
  let { lora_name, strength_model, strength_clip } = vals;

  if (!reverse) {
    if (oldType === "LoraLoaderModelOnly" || strength_clip === undefined) {
      strength_clip = strength_model;
    }
    applyWidgetValues(newNode, oldNode, { lora_name, strength_model, strength_clip }, false);
  } else {
    const isModelOnly = targetType === "LoraLoaderModelOnly";
    applyWidgetValues(newNode, oldNode, { lora_name, strength_model, strength_clip }, isModelOnly);
  }

  if (Array.isArray(newNode.inputs)) {
    for (let i = 0; i < newNode.inputs.length; i++) {
      const newInp = newNode.inputs[i];
      const oldInp = oldNode.inputs?.find((inp) => inp.name.toLowerCase() === newInp.name.toLowerCase()) || oldNode.inputs?.[i];
      if (oldInp) {
        Object.assign(newInp, oldInp);
      }
      newInp.link = (oldInp && oldInp.link !== null && oldInp.link !== undefined) ? oldInp.link : null;
    }
  }

  if (Array.isArray(newNode.outputs)) {
    for (let i = 0; i < newNode.outputs.length; i++) {
      const newOut = newNode.outputs[i];
      const oldOut = oldNode.outputs?.find((out) => out.name.toLowerCase() === newOut.name.toLowerCase()) || oldNode.outputs?.[i];
      if (oldOut) {
        Object.assign(newOut, oldOut);
      }
      newOut.links = (oldOut && oldOut.links) ? (Array.isArray(oldOut.links) ? [...oldOut.links] : oldOut.links) : null;
    }
  }

  if (Array.isArray(graph._nodes)) {
    const idx = graph._nodes.indexOf(oldNode);
    if (idx !== -1) graph._nodes[idx] = newNode;
    else {
      const idIdx = graph._nodes.findIndex((n) => n.id === oldNode.id);
      if (idIdx !== -1) graph._nodes[idIdx] = newNode;
      else graph._nodes.push(newNode);
    }
  }
  if (Array.isArray(graph.nodes)) {
    const idx = graph.nodes.indexOf(oldNode);
    if (idx !== -1) graph.nodes[idx] = newNode;
    else {
      const idIdx = graph.nodes.findIndex((n) => n.id === oldNode.id);
      if (idIdx !== -1) graph.nodes[idIdx] = newNode;
      else graph.nodes.push(newNode);
    }
  }
  if (graph._nodes_by_id) {
    graph._nodes_by_id[newNode.id] = newNode;
  }
  newNode.graph = graph;

  if (canvas?.selected_nodes) {
    if (Array.isArray(canvas.selected_nodes)) {
      const sidx = canvas.selected_nodes.indexOf(oldNode);
      if (sidx !== -1) canvas.selected_nodes[sidx] = newNode;
    } else if (canvas.selected_nodes[oldNode.id]) {
      canvas.selected_nodes[newNode.id] = newNode;
    }
  }

  oldNode.onRemoved?.();
  newNode.onAdded?.(graph);
  newNode.setDirtyCanvas?.(true, true);
}

export function convertLoraLoaders(reverse = false, canvas = null, clickedNode = null, explicitGraph = null) {
  const graph = explicitGraph || app.graph || canvas?.graph;
  if (!graph) return false;

  if (!reverse) {
    if (!isNodeTypeRegistered("KiLoRALoRALoader")) {
      showNotification("KiLoRALoRALoader node type is not registered.");
      return false;
    }
  } else {
    if (typeof LiteGraph !== "undefined" && LiteGraph.registered_node_types) {
      if (!LiteGraph.registered_node_types["LoraLoader"] && !LiteGraph.registered_node_types["LoraLoaderModelOnly"]) {
        showNotification("Built-in LoRA loader node types are not registered.");
        return false;
      }
    }
  }

  const isCandidate = (n) => {
    const type = n?.type || n?.comfyClass;
    if (!reverse) {
      return type === "LoraLoader" || type === "LoraLoaderModelOnly";
    }
    return type === "KiLoRALoRALoader";
  };

  const allNodes = graph._nodes || graph.nodes || [];
  const selected = getSelectedNodes(canvas);
  const selectedMatching = selected.filter(isCandidate);

  let targets = [];
  if (selectedMatching.length > 0) {
    targets = selectedMatching;
  } else if (clickedNode && isCandidate(clickedNode)) {
    targets = [clickedNode];
  } else if (selected.length === 0) {
    targets = allNodes.filter(isCandidate);
  } else {
    return false;
  }

  if (targets.length === 0) return false;

  graph.beforeChange?.();
  try {
    for (const oldNode of targets) {
      replaceNode(oldNode, reverse, graph, canvas);
    }
  } finally {
    graph.afterChange?.();
    graph.setDirtyCanvas?.(true, true);
    canvas?.setDirty?.(true, true);
  }
  return true;
}

function setupCanvasAndNodeMenus() {
  if (typeof LGraphCanvas !== "undefined" && LGraphCanvas.prototype) {
    const origCanvasMenu = LGraphCanvas.prototype.getCanvasMenuOptions;
    LGraphCanvas.prototype.getCanvasMenuOptions = function (options) {
      const r = origCanvasMenu?.apply(this, arguments);
      const opts = options || arguments[0] || [];
      if (!opts.some((opt) => opt?.content === "KiLoRA: convert LoRA loaders")) {
        opts.push(
          null,
          {
            content: "KiLoRA: convert LoRA loaders",
            callback: () => convertLoraLoaders(false, this),
          },
          {
            content: "KiLoRA: convert back to LoRA loaders",
            callback: () => convertLoraLoaders(true, this),
          }
        );
      }
      return r;
    };
  }

  if (typeof LGraphNode !== "undefined" && LGraphNode.prototype) {
    const origNodeMenu = LGraphNode.prototype.getExtraMenuOptions;
    LGraphNode.prototype.getExtraMenuOptions = function (canvas, options) {
      const r = origNodeMenu?.apply(this, arguments);
      const opts = options || arguments[1] || [];
      if (!opts.some((opt) => opt?.content === "KiLoRA: convert LoRA loaders")) {
        opts.push(
          null,
          {
            content: "KiLoRA: convert LoRA loaders",
            callback: () => convertLoraLoaders(false, canvas, this),
          },
          {
            content: "KiLoRA: convert back to LoRA loaders",
            callback: () => convertLoraLoaders(true, canvas, this),
          }
        );
      }
      return r;
    };
  }
}

app.registerExtension({
  name: "harnessix.kilora",

  async setup() {
    await loadStrings();
    app.ui?.settings?.addSetting?.({
      id: "KiLoRA.PrepareAtStartup",
      name: t("prepare.startup"),
      type: "boolean",
      defaultValue: true,
      onChange: async (enabled) => {
        await api.fetchApi("/kilora/prepare-startup", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ enabled }),
        });
      },
    });
    setupCanvasAndNodeMenus();
  },

  getCanvasMenuItems(canvas) {
    return [
      {
        content: "KiLoRA: convert LoRA loaders",
        callback: () => convertLoraLoaders(false, canvas),
      },
      {
        content: "KiLoRA: convert back to LoRA loaders",
        callback: () => convertLoraLoaders(true, canvas),
      },
    ];
  },

  getNodeMenuItems(node) {
    return [
      {
        content: "KiLoRA: convert LoRA loaders",
        callback: () => convertLoraLoaders(false, app.canvas, node),
      },
      {
        content: "KiLoRA: convert back to LoRA loaders",
        callback: () => convertLoraLoaders(true, app.canvas, node),
      },
    ];
  },

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!nodeData?.name?.startsWith(PREFIX)) return;
    const isLive = LIVE_NODES.has(nodeData.name);

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated?.apply(this, arguments);
      this.color = COLORS.title;
      this.bgcolor = COLORS.body;
      if (isLive) this.boxcolor = COLORS.liveEdge;
      return r;
    };

    const onDrawForeground = nodeType.prototype.onDrawForeground;
    nodeType.prototype.onDrawForeground = function (ctx) {
      const r = onDrawForeground?.apply(this, arguments);
      if (this.flags?.collapsed) return r;
      drawBadge(ctx, this);
      if (isLive) drawLiveState(ctx, this);
      return r;
    };

    if (isLive) {
      const getExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
      nodeType.prototype.getExtraMenuOptions = function (canvas, options) {
        const r = getExtraMenuOptions?.apply(this, arguments);
        options.unshift({ content: t("web.menu.open"), callback: () => showPopover(this) }, null);
        return r;
      };
    }

    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      const r = onExecuted?.apply(this, arguments);
      const text = (message?.text || []).join("\n");
      if (text) {
        const prev = this.kiloraLive || {};
        const next = { ...prev, ...parseReport(text) };
        if (LIVE_NODES.has(this.type)) {
          const hist = (prev.history || []).concat([{ i: (prev.runs || 0) + 1, w: next.writes, u: next.undo }]).slice(-20);
          next.history = hist; next.runs = (prev.runs || 0) + 1;
        }
        this.kiloraLive = next;
        this.setDirtyCanvas?.(true, true);
      }
      return r;
    };
  },
});
