// Copyright The KiLoRA Authors
// SPDX-License-Identifier: Apache-2.0
import { readFileSync, readdirSync, existsSync } from "node:fs";
import { execFileSync } from "node:child_process";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import assert from "node:assert";

const WEB = join(dirname(fileURLToPath(import.meta.url)), "..");
const REPO = join(WEB, "..");
let failures = 0;
const fail = (msg) => { failures++; console.log("FAIL " + msg); };
const pass = (msg) => console.log("ok   " + msg);

const dir = join(WEB, "i18n");
const files = readdirSync(dir).filter((f) => f.endsWith(".json")).sort();
const en = JSON.parse(readFileSync(join(dir, "en.json"), "utf8"));
const holes = (s) => [...s.matchAll(/\{(\w+)\}/g)].map((m) => m[1]).sort().join(",");
for (const f of files) {
  const data = JSON.parse(readFileSync(join(dir, f), "utf8"));
  const missing = Object.keys(en).filter((k) => !(k in data));
  const extra = Object.keys(data).filter((k) => !(k in en));
  const badHoles = Object.keys(en).filter((k) => k in data && holes(en[k]) !== holes(data[k]));
  const empty = Object.keys(data).filter((k) => !String(data[k]).trim());
  if (missing.length || extra.length || badHoles.length || empty.length)
    fail(`${f}: missing ${missing}, extra ${extra}, placeholder mismatch ${badHoles}, empty ${empty}`);
}
pass(`i18n parity: ${files.length} locale files, ${Object.keys(en).length} keys each`);

const js = readFileSync(join(WEB, "kilora.js"), "utf8");
const jsLocales = JSON.parse(js.match(/const LOCALES = (\[[^\]]+\])/)[1].replace(/'/g, '"'));
const fileLocales = files.map((f) => f.replace(/\.json$/, ""));
const same = (a, b) => [...a].sort().join() === [...b].sort().join();
if (!same(jsLocales, fileLocales)) fail(`kilora.js LOCALES ${jsLocales} differ from web/i18n files ${fileLocales}`);
const py = join(REPO, "src/kilora/i18n/__init__.py");
if (existsSync(py)) {
  const block = readFileSync(py, "utf8").match(/LOCALES = \(([\s\S]*?)\)/);
  const pyLocales = block ? [...block[1].matchAll(/"([^"]+)"/g)].map((m) => m[1]) : [];
  if (!same(pyLocales, fileLocales)) fail(`node-side LOCALES ${pyLocales} differ from web/i18n ${fileLocales}`);
  else pass(`locale list matches the node side (${pyLocales.length} locales)`);
} else {
  console.log("skip node-side locale list: src/kilora/i18n/__init__.py not found");
}

for (const f of ["kilora.js", "tests/run.mjs"]) {
  try { execFileSync(process.execPath, ["--check", join(WEB, f)], { stdio: "pipe" }); }
  catch (e) { fail(`syntax ${f}: ${e.stderr}`); }
}
const LINT = [
  [/\bvar\s/, "use const or let, not var"],
  [/[^=!]==[^=]/, "use === and !=="],
  [/!=[^=]/, "use !=="],
  [/\beval\s*\(|new Function\s*\(/, "no eval or new Function"],
  [/console\.(log|debug)\(/, "no console.log or console.debug"],
  [/debugger;/, "no debugger statements"],
  [/fetch\(\s*["'`]https?:/, "no requests to other hosts"],
];
const lines = js.split("\n");
lines.forEach((line, i) => {
  const code = line.replace(/\/\/.*$/, "").replace(/(["'`])(?:\\.|(?!\1).)*\1/g, '""');
  for (const [re, msg] of LINT) if (re.test(code)) fail(`kilora.js:${i + 1}: ${msg}`);
  if (/\.innerHTML\s*=/.test(code) && !/svg/i.test(line)) fail(`kilora.js:${i + 1}: innerHTML only for constant SVG icons`);
});
pass(`lint: kilora.js (${lines.length} lines) and the test runner`);

const start = js.indexOf("const PATTERNS");
const end = js.indexOf("function learningFromWidget");
const parseReport = new Function(js.slice(start, end) + "; return parseReport;")();
const cases = [
  ["inkwash: 3 undo entries, learned strength 1", { undo: 3 }],
  ["Live write done. 12 writes since reset, undo depth: 4", { undo: 4, writes: 12 }],
  ["Learning is off", { learning: false }],
  ["Paused. writes: 7", { writes: 7, learning: false }],
  ["Reset: the learned change is empty (undo brings it back).", { writes: 0 }],
  ["Loaded inkwash.kmod", {}],
];
for (const [text, want] of cases) {
  const got = parseReport(text);
  if (JSON.stringify(got) !== JSON.stringify(want)) fail(`parseReport(${JSON.stringify(text)}) = ${JSON.stringify(got)}, want ${JSON.stringify(want)}`);
}
pass(`report parser: ${cases.length} cases`);

const TPL = join(REPO, "example_workflows");
const known = new Set(Object.keys(JSON.parse(readFileSync(join(REPO, "node_list.json"), "utf8"))));
const tplFiles = existsSync(TPL) ? readdirSync(TPL, { withFileTypes: true }) : [];
if (!tplFiles.some((f) => f.isFile() && f.name.endsWith(".json"))) fail("example_workflows/ has no *.json template");
for (const f of tplFiles) {
  if (f.isDirectory()) fail(`example_workflows/${f.name}/: the template browser does not scan subfolders`);
  if (!f.isFile() || !f.name.endsWith(".json")) continue;
  let g;
  try { g = JSON.parse(readFileSync(join(TPL, f.name), "utf8")); } catch (e) { fail(`${f.name}: not JSON (${e.message})`); continue; }
  if (!Array.isArray(g.nodes) || !Array.isArray(g.links)) { fail(`${f.name}: not a UI-format workflow (nodes and links arrays)`); continue; }
  const ids = new Set(g.nodes.map((n) => n.id));
  if (ids.size !== g.nodes.length) fail(`${f.name}: duplicate node ids`);
  for (const n of g.nodes) {
    if (typeof n.type !== "string") fail(`${f.name}: node ${n.id} has no type`);
    else if (n.type.startsWith("KiLoRA") && !known.has(n.type)) fail(`${f.name}: node ${n.id} type ${n.type} is not in node_list.json`);
  }
  const linkIds = new Set();
  for (const l of g.links) {
    const [id, from, , to] = l;
    linkIds.add(id);
    if (!ids.has(from) || !ids.has(to)) fail(`${f.name}: link ${id} joins a missing node`);
  }
  for (const n of g.nodes) for (const inp of n.inputs || []) if (inp.link != null && !linkIds.has(inp.link)) fail(`${f.name}: node ${n.id} input ${inp.name} names missing link ${inp.link}`);
  if (g.last_node_id != null && g.last_node_id < Math.max(...ids)) fail(`${f.name}: last_node_id is below the largest node id`);
}
pass(`templates: ${tplFiles.filter((f) => f.name.endsWith(".json")).length} files in example_workflows/`);

const convStart = js.indexOf("// --- LoRA loader converter");
const convEnd = js.indexOf("// --- End LoRA loader converter");
if (convStart === -1 || convEnd === -1) {
  fail("could not find LoRA loader converter code markers in kilora.js");
} else {
  let lastNotification = null;
  const mockApp = {
    graph: null,
    canvas: null,
    extensionManager: {
      toast: {
        add(opts) { lastNotification = opts.detail; },
        addAlert(msg) { lastNotification = msg; },
      },
    },
  };
  const mockLiteGraph = {
    registered_node_types: {},
  };

  const makeConverter = new Function("app", "LiteGraph", "alert",
    js.slice(convStart, convEnd).replace(/\bexport\s+/g, "") + "\nreturn { convertLoraLoaders, isNodeTypeRegistered };");
  const { convertLoraLoaders } = makeConverter(mockApp, mockLiteGraph, (m) => { lastNotification = m; });

  const buildTestGraph = () => {
    const nodes = [
      {
        id: 1,
        type: "CheckpointLoaderSimple",
        pos: [0, 0],
        size: [200, 100],
        title: "Checkpoint",
        inputs: [],
        outputs: [
          { name: "MODEL", type: "MODEL", links: [1] },
          { name: "CLIP", type: "CLIP", links: [2] },
        ],
        widgets_values: ["v1-5-pruned.safetensors"],
      },
      {
        id: 2,
        type: "LoraLoader",
        pos: [250, 0],
        size: [200, 120],
        title: "LoRA 1",
        inputs: [
          { name: "model", type: "MODEL", link: 1 },
          { name: "clip", type: "CLIP", link: 2 },
        ],
        outputs: [
          { name: "MODEL", type: "MODEL", links: [3] },
          { name: "CLIP", type: "CLIP", links: [4] },
        ],
        widgets_values: ["lora_a.safetensors", 0.8, 0.7],
      },
      {
        id: 3,
        type: "LoraLoader",
        pos: [500, 0],
        size: [200, 120],
        title: "LoRA 2",
        inputs: [
          { name: "model", type: "MODEL", link: 3 },
          { name: "clip", type: "CLIP", link: 4 },
        ],
        outputs: [
          { name: "MODEL", type: "MODEL", links: [5] },
          { name: "CLIP", type: "CLIP", links: [6] },
        ],
        widgets_values: ["lora_b.safetensors", 0.5, 0.6],
      },
      {
        id: 4,
        type: "LoraLoaderModelOnly",
        pos: [750, 0],
        size: [200, 100],
        title: "LoRA Model Only",
        inputs: [
          { name: "model", type: "MODEL", link: 5 },
        ],
        outputs: [
          { name: "MODEL", type: "MODEL", links: [7] },
        ],
        widgets_values: ["lora_c.safetensors", 0.9],
      },
      {
        id: 5,
        type: "CLIPTextEncode",
        pos: [750, 200],
        size: [200, 100],
        title: "Prompt",
        inputs: [
          { name: "clip", type: "CLIP", link: 6 },
        ],
        outputs: [{ name: "CONDITIONING", type: "CONDITIONING", links: [] }],
        widgets_values: ["masterpiece"],
      },
      {
        id: 6,
        type: "KSampler",
        pos: [1000, 0],
        size: [200, 200],
        title: "KSampler",
        inputs: [
          { name: "model", type: "MODEL", link: 7 },
        ],
        outputs: [{ name: "LATENT", type: "LATENT", links: [] }],
        widgets_values: [0, "randomize", 20, 8, "euler", "normal", 1],
      },
    ];

    const links = [
      [1, 1, 0, 2, 0, "MODEL"],
      [2, 1, 1, 2, 1, "CLIP"],
      [3, 2, 0, 3, 0, "MODEL"],
      [4, 2, 1, 3, 1, "CLIP"],
      [5, 3, 0, 4, 0, "MODEL"],
      [6, 3, 1, 5, 0, "CLIP"],
      [7, 4, 0, 6, 0, "MODEL"],
    ];

    return {
      nodes: JSON.parse(JSON.stringify(nodes)),
      _nodes: JSON.parse(JSON.stringify(nodes)),
      links: JSON.parse(JSON.stringify(links)),
      beforeChangeCount: 0,
      afterChangeCount: 0,
      beforeChange() { this.beforeChangeCount++; },
      afterChange() { this.afterChangeCount++; },
    };
  };

  const unregGraph = buildTestGraph();
  const unregSnapshot = JSON.stringify(unregGraph.nodes);
  lastNotification = null;
  const unregResult = convertLoraLoaders(false, null, null, unregGraph);
  if (unregResult !== false) fail("convertLoraLoaders should return false when KiLoRALoRALoader is not registered");
  if (!lastNotification || !/not registered/i.test(lastNotification)) fail(`expected unregistered message, got "${lastNotification}"`);
  if (JSON.stringify(unregGraph.nodes) !== unregSnapshot) fail("graph was modified when KiLoRALoRALoader was not registered");

  mockLiteGraph.registered_node_types = {
    KiLoRALoRALoader: class {},
    LoraLoader: class {},
    LoraLoaderModelOnly: class {},
  };

  const graph = buildTestGraph();
  const initialNodes = JSON.parse(JSON.stringify(graph.nodes));
  const initialLinks = JSON.parse(JSON.stringify(graph.links));

  const forwardOk = convertLoraLoaders(false, null, null, graph);
  if (!forwardOk) fail("forward conversion returned false");
  if (graph.beforeChangeCount !== 1 || graph.afterChangeCount !== 1) {
    fail(`expected 1 undo step, got beforeChange=${graph.beforeChangeCount}, afterChange=${graph.afterChangeCount}`);
  }

  const n2 = graph.nodes.find((n) => n.id === 2);
  const n3 = graph.nodes.find((n) => n.id === 3);
  const n4 = graph.nodes.find((n) => n.id === 4);
  if (n2?.type !== "KiLoRALoRALoader" || n3?.type !== "KiLoRALoRALoader" || n4?.type !== "KiLoRALoRALoader") {
    fail(`nodes 2, 3, 4 should be KiLoRALoRALoader, got ${n2?.type}, ${n3?.type}, ${n4?.type}`);
  }
  if (n4?.widgets_values?.[2] !== n4?.widgets_values?.[1]) {
    fail(`model-only node 4 should have strength_clip (${n4?.widgets_values?.[2]}) equal to strength_model (${n4?.widgets_values?.[1]})`);
  }
  if (n4?.inputs?.[0]?.link !== 5 || n4?.inputs?.[1]?.link !== null) {
    fail("model-only node 4 should keep model link and have clip input link as null");
  }

  const reverseOk = convertLoraLoaders(true, null, null, graph);
  if (!reverseOk) fail("reverse conversion returned false");
  if (graph.beforeChangeCount !== 2 || graph.afterChangeCount !== 2) {
    fail(`expected 2 undo steps total, got beforeChange=${graph.beforeChangeCount}, afterChange=${graph.afterChangeCount}`);
  }

  for (const n of graph.nodes) delete n.graph;
  for (const n of graph._nodes) delete n.graph;
  delete graph.beforeChangeCount;
  delete graph.afterChangeCount;
  delete graph.beforeChange;
  delete graph.afterChange;

  try {
    assert.deepStrictEqual(graph.nodes, initialNodes);
    assert.deepStrictEqual(graph.links, initialLinks);
  } catch (err) {
    fail(`graph after convert and convert back did not equal the start: ${err.message}`);
  }

  const selGraph = buildTestGraph();
  const mockCanvas = {
    selected_nodes: { 2: selGraph.nodes[1] },
  };
  convertLoraLoaders(false, mockCanvas, null, selGraph);
  if (selGraph.nodes[1].type !== "KiLoRALoRALoader") fail("selected node 2 should be converted to KiLoRALoRALoader");
  if (selGraph.nodes[2].type !== "LoraLoader") fail("unselected node 3 should remain LoraLoader");
  if (selGraph.nodes[3].type !== "LoraLoaderModelOnly") fail("unselected node 4 should remain LoraLoaderModelOnly");

  const liveGraph = {
    nodes: [
      {
        id: 10,
        type: "LoraLoaderModelOnly",
        pos: [100, 100],
        title: "Live Model Only",
        inputs: [{ name: "model", type: "MODEL", link: null }],
        outputs: [{ name: "MODEL", type: "MODEL", links: null }],
        widgets: [
          { name: "lora_name", value: "live.safetensors" },
          { name: "strength_model", value: 0.75 },
        ],
      },
    ],
    links: [],
  };
  liveGraph._nodes = liveGraph.nodes;
  const initialLiveNodes = JSON.parse(JSON.stringify(liveGraph.nodes));
  convertLoraLoaders(false, null, null, liveGraph);
  const liveKiLoRA = liveGraph.nodes[0];
  if (liveKiLoRA.type !== "KiLoRALoRALoader") fail("live node should be KiLoRALoRALoader");
  const clipW = liveKiLoRA.widgets.find((w) => w.name === "strength_clip");
  if (clipW?.value !== 0.75) fail(`live KiLoRA strength_clip should be 0.75, got ${clipW?.value}`);
  convertLoraLoaders(true, null, null, liveGraph);
  delete liveGraph.nodes[0].graph;
  delete liveGraph._nodes[0].graph;
  try {
    assert.deepStrictEqual(liveGraph.nodes, initialLiveNodes);
  } catch (err) {
    fail(`live node after reverse did not match initial: ${err.message}`);
  }

  pass("LoRA loader converter: chained + model-only converted and reversed equal to start, unregistered check, selection, and undo");
}

console.log(failures ? `\n${failures} failure(s)` : "\nall front-end checks passed");
process.exit(failures ? 1 : 0);
