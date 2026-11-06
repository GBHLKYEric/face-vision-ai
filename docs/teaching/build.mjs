// 把 parts/*.md 构建成单页 HTML (样式与「变压器与量子物理详解」等页面一致)。
// 用法: node docs/teaching/build.mjs   (需要 katex 和 marked: npm i katex marked)
// Markdown 约定:
//   ## 专题三|人脸检测|副标题 {#t3|三 检测}      一个顶部标签
//   ### 标题 {#t3-s1}                           小节 (进第二行标签)
//   #### 例题 1|小标题 {#t3-e1}                  一道例题/练习 (article.q)
//   **1. 标题。** 正文 {#t3-k1}                  知识要点
//   ```latex  公式 (KaTeX 预渲染)     ```python 代码
//   以「提醒：」开头 -> 黄色; 「口诀：/易错点：/类比：/验算：/结论：/记忆方法：」开头 -> 绿色标签
//   <figure>/<div>/<svg> 开头的块原样输出
import fs from "fs";
import path from "path";
import { createRequire } from "module";

const here = path.dirname(new URL(import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, "$1"));
const require = createRequire(import.meta.url);
const NODE_PATHS = [here, path.join(here, "../../_scratch/teach")];
function req(name) {
  for (const p of NODE_PATHS) {
    try { return require(require.resolve(name, { paths: [p] })); } catch (e) { /* try next */ }
  }
  return require(name);
}
const katex = req("katex");
const { marked } = req("marked");
const KDIR = path.dirname(path.dirname(require.resolve("katex", { paths: NODE_PATHS })));

// 内容按章节拆在 parts/ 目录下的多个 .md 里, 按文件名顺序拼接
const partDir = path.join(here, "parts");
// 实验数字由 tools/collect_results.py 汇总到 results.json, 这里替换 {{占位符}}, 避免手抄出错
const resPath = path.join(here, "results.json");
const results = fs.existsSync(resPath) ? JSON.parse(fs.readFileSync(resPath, "utf8")) : {};
const src = fs.readdirSync(partDir).filter(f => f.endsWith(".md")).sort()
  .map(f => fs.readFileSync(path.join(partDir, f), "utf8")).join("\n\n").replace(/\r\n/g, "\n")
  .replace(/\{\{(\w+)\}\}/g, (m, k) => (k in results ? results[k] : m));
const lines = src.split("\n");

let eqCount = 0, latexBlocks = 0;
const usedFonts = new Set(["KaTeX_Main-Regular", "KaTeX_Main-Bold", "KaTeX_Main-Italic", "KaTeX_Math-Italic",
  "KaTeX_AMS-Regular", "KaTeX_Size1-Regular", "KaTeX_Size2-Regular", "KaTeX_Size3-Regular", "KaTeX_Size4-Regular"]);

function tex(s) {
  latexBlocks++;
  const html = katex.renderToString(s, { displayMode: true, throwOnError: true, strict: "error", output: "htmlAndMathml" });
  eqCount++;
  if (/mathcal/.test(s)) usedFonts.add("KaTeX_Caligraphic-Regular");
  if (/mathsf/.test(s)) usedFonts.add("KaTeX_SansSerif-Regular");
  if (/mathtt|texttt/.test(s)) usedFonts.add("KaTeX_Typewriter-Regular");
  if (/boldsymbol|mathbf/.test(s)) { usedFonts.add("KaTeX_Math-BoldItalic"); usedFonts.add("KaTeX_Main-Bold"); }
  return `<div class="eq">${html}</div>`;
}

const esc = s => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
function code(lang, body) {
  let h = esc(body);
  if (lang === "python" || lang === "py") {
    // 极简高亮: 注释、字符串、关键字
    h = h.replace(/(#[^\n]*)/g, '<span class="cm">$1</span>')
      .replace(/\b(def|class|return|for|in|if|else|elif|import|from|with|as|while|not|and|or|None|True|False|lambda|yield)\b(?![^<]*<\/span>)/g, '<span class="kw">$1</span>');
  }
  return `<pre class="code"><code>${h}</code></pre>`;
}
const inline = s => marked.parseInline(s);

function idOf(text) {
  const m = /\s*\{#([^}|]+)(?:\|([^}]*))?\}\s*$/.exec(text);
  if (!m) return [text, null, null];
  return [text.slice(0, m.index), m[1], m[2] || null];
}

const sections = []; // {id, tab, parts:[{id,label}], html:[]}
let cur = null, inQ = false;
const out = () => cur.html;
function closeQ() { if (inQ) { out().push("</article>"); inQ = false; } }

const TIP = /^(口诀|易错点|类比|验算|结论|记忆方法|判断口诀|面试答法|小结)：/;
let i = 0;
let para = [];
function flushPara() {
  if (!para.length) return;
  const text = para.join("\n");
  para = [];
  if (!cur) return;
  let m;
  if (/^\s*<(figure|div|svg|table|details)/.test(text)) { out().push(text); return; }
  if ((m = /^\*\*(\d+)\. ([^*]+)\*\*\s*([\s\S]*)$/.exec(text))) {
    const [body, id] = idOf(m[3]);
    out().push(`<p class="kp"${id ? ` id="${id}"` : ""}><span class="mn">要点 ${m[1]}</span><b>${inline(m[2])}</b> ${inline(body)}</p>`);
    return;
  }
  if ((m = /^提醒：([\s\S]*)$/.exec(text))) { out().push(`<p class="flag"><span class="lab">注意</span>${inline(m[1])}</p>`); return; }
  if ((m = TIP.exec(text))) { out().push(`<p class="tip"><span class="lab">${m[1]}</span>${inline(text.slice(m[0].length))}</p>`); return; }
  if ((m = /^> ([\s\S]*)$/.exec(text)) && out().length && /<h2/.test(out()[out().length - 1])) {
    out().push(`<p class="lead">${inline(m[1])}</p>`); return;
  }
  out().push(marked.parse(text));
}

while (i < lines.length) {
  const line = lines[i];
  let m;
  if ((m = /^```(\w*)\s*$/.exec(line))) {
    flushPara();
    const lang = m[1];
    const body = [];
    i++;
    while (i < lines.length && !/^```\s*$/.test(lines[i])) body.push(lines[i++]);
    i++;
    out().push(lang === "latex" ? tex(body.join("\n")) : code(lang, body.join("\n")));
    continue;
  }
  if ((m = /^## (.*)$/.exec(line))) {
    flushPara(); if (cur) closeQ();
    const [t, id, tab] = idOf(m[1]);
    const [no, title, small] = t.split("|").map(s => s.trim());
    cur = { id, tab: tab || title, parts: [], html: [] };
    sections.push(cur);
    if (id === "home") cur.html.push(`<header class="hero"><h1>${inline(title)}</h1>`);
    else cur.html.push(`<h2><span class="no">${no}</span>${inline(title)}${small ? `<small>${inline(small)}</small>` : ""}</h2>`);
    i++; continue;
  }
  if ((m = /^### (.*)$/.exec(line))) {
    flushPara(); closeQ();
    const [t, id, short] = idOf(m[1]);
    const [k, title] = t.includes("|") ? t.split("|").map(s => s.trim()) : [null, t.trim()];
    if (cur.id === "home" && cur.html[cur.html.length - 1] !== "</header>" && !cur.html.includes("</header>")) cur.html.push("</header>");
    cur.parts.push({ id, label: short || title });
    out().push(`<h3 id="${id}">${k ? `<span class="k">${k}</span>` : ""}${inline(title)}</h3>`);
    i++; continue;
  }
  if ((m = /^#### (.*)$/.exec(line))) {
    flushPara(); closeQ();
    const [t, id] = idOf(m[1]);
    const [qn, ttl] = t.split("|").map(s => s.trim());
    out().push(`<article class="q" id="${id}"><h4><span class="qn">${inline(qn)}</span>${ttl ? `<span class="ttl">${inline(ttl)}</span>` : ""}</h4>`);
    inQ = true; i++; continue;
  }
  if (line.trim() === "") { flushPara(); i++; continue; }
  para.push(line); i++;
}
flushPara(); closeQ();
for (const s of sections) if (s.id === "home" && !s.html.includes("</header>")) s.html.push("</header>");

// ---- KaTeX CSS with inlined fonts ----
let kcss = fs.readFileSync(path.join(KDIR, "dist/katex.min.css"), "utf8");
kcss = kcss.replace(/@font-face\{[^}]*\}/g, rule => {
  const m = /url\(fonts\/([\w-]+)\.woff2\)/.exec(rule);
  if (!m || !usedFonts.has(m[1])) return "";
  const b64 = fs.readFileSync(path.join(KDIR, "dist/fonts", m[1] + ".woff2")).toString("base64");
  return rule.replace(/src:[^}]*/, `src:url(data:font/woff2;base64,${b64}) format("woff2")`);
});

const pageCss = fs.readFileSync(path.join(here, "page.css"), "utf8");
const pageJs = fs.readFileSync(path.join(here, "page.js"), "utf8");
const nav = [
  `<nav class="top" aria-label="目录"><div class="row tabs">`,
  sections.map((s, k) => `<a href="#${s.id}"${k === 0 ? ' aria-current="true"' : ""}>${s.tab}</a>`).join(""),
  `</div>`,
  ...sections.map((s, k) => `<div class="row parts" data-for="${s.id}"${k ? " hidden" : ""}>` +
    s.parts.map(p => `<a href="#${p.id}">${p.label}</a>`).join("") + `</div>`),
  `</nav>`,
].join("");

const titleLine = /^## [^|]*\|([^|{]*)/.exec(src.split("\n").find(l => l.startsWith("## ")))[1].trim();
const head = `<title>${titleLine}</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Noto+Serif+SC:wght@600;700&display=swap">
<style>${kcss}</style>
<style>${pageCss}</style>`;
const body = `${nav}
<div class="wrap">
${sections.map(s => `<section class="sec" id="${s.id}">\n${s.html.join("\n")}\n</section>`).join("\n")}
</div>
<script>${pageJs}</script>`;
// index.html 可以直接用浏览器打开; artifact.html 发布时由平台套上 <html>/<head> 外壳
const html = `<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
${head}</head><body>
${body}
</body></html>`;
fs.writeFileSync(path.join(here, "artifact.html"), `${head}\n${body}\n`);

// ---- 检查: 重复 id / 失效链接 ----
const ids = [...html.matchAll(/\sid="([^"]+)"/g)].map(m => m[1]);
const dup = ids.filter((x, k) => ids.indexOf(x) !== k);
const links = [...html.matchAll(/href="#([^"]+)"/g)].map(m => m[1]);
const broken = [...new Set(links.filter(l => !ids.includes(l)))];
fs.writeFileSync(path.join(here, "index.html"), html);
console.log(`sections=${sections.length} latex=${latexBlocks} rendered=${eqCount} size=${(html.length / 1024).toFixed(0)}KB`);
if (dup.length) console.log("重复 id:", [...new Set(dup)]);
if (broken.length) console.log("失效链接:", broken);
const todo = [...src.matchAll(/\{\{(\w+)\}\}/g)].map(m => m[1]);
if (todo.length) console.log("未填写的占位符:", todo);
