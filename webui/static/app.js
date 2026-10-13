"use strict";

/* ================= Constants ================= */
const EMOTIONS = ["Surprise", "Fear", "Disgust", "Happy", "Sad", "Anger", "Neutral"];
const EMO_COLORS = ["#ff9248", "#a77bff", "#7fcf5a", "#ffd23f", "#4d9fff", "#ff5468", "#9aa3b5"];
const MODEL_COLORS = { efficientnet: "#7c6cff", densenet: "#2fbfa0", resnet: "#ff9f43" };
const MODEL_LABELS = { efficientnet: "EfficientNet-B0", densenet: "DenseNet-121", resnet: "ResNet-18" };
const MODEL_ORDER = ["efficientnet", "densenet", "resnet"];

// Numbers from the README (RAF-DB, current 20-epoch models in Video Predicition/)
const BENCH = {
  models: {
    resnet: { params: "11.2M", size: "45 MB", epoch: "2.1 min", acc: 87.22 },
    densenet: { params: "7.0M", size: "28 MB", epoch: "5.0 min", acc: 88.01 },
    efficientnet: { params: "4.0M", size: "16 MB", epoch: "3.6 min", acc: 87.06 },
  },
  perClass: { // Surprise, Fear, Disgust, Happy, Sad, Anger, Neutral
    resnet: [87.8, 56.8, 62.5, 93.8, 87.9, 80.9, 85.6],
    densenet: [86.9, 58.1, 60.6, 94.9, 85.4, 85.2, 88.8],
    efficientnet: [88.1, 58.1, 65.0, 92.8, 89.1, 79.0, 85.3],
  },
  testCounts: [329, 74, 160, 1185, 478, 162, 680],
  trainCounts: [1290, 281, 717, 4772, 1982, 705, 2524],
  aug: { // 12-epoch runs: baseline_weights/ -> aug12_weights/
    conditions: ["Clean", "Dark", "Low contrast", "Occluded", "Tight crop"],
    resnet: [[86.9, 87.0], [85.3, 86.9], [84.7, 86.7], [80.5, 84.2], [73.9, 83.4]],
    densenet: [[87.5, 87.5], [85.1, 87.0], [85.1, 87.2], [80.0, 85.0], [79.3, 84.8]],
    efficientnet: [[87.4, 87.1], [85.2, 87.1], [85.3, 87.2], [82.8, 85.4], [82.7, 85.2]],
  },
};

/* ================= Helpers ================= */
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const sleep = ms => new Promise(r => setTimeout(r, ms));
const pct = (v, d = 1) => (v * 100).toFixed(d) + "%";
const esc = s => String(s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const argmax = a => a.reduce((bi, v, i) => (v > a[bi] ? i : bi), 0);
const SVGNS = "http://www.w3.org/2000/svg";

function toast(msg, isErr = false) {
  const t = $("#toast");
  t.textContent = msg;
  t.className = "toast" + (isErr ? " err" : "");
  t.hidden = false;
  clearTimeout(toast._t);
  toast._t = setTimeout(() => (t.hidden = true), 3500);
}

function faceSVG(idx, size = 54) {
  const c = EMO_COLORS[idx] ?? "#6b7285";
  const mouths = {
    0: '<ellipse cx="32" cy="44" rx="6" ry="7.5" fill="#1a1a1f" stroke="none"/>',
    1: '<path d="M22 46q5-5 10 0t10 0"/>',
    2: '<path d="M22 44q6-6 12 0q4 3 8-1"/>',
    3: '<path d="M20 38q12 14 24 0"/>',
    4: '<path d="M22 47q10-9 20 0"/>',
    5: '<path d="M22 46h20"/>',
    6: '<path d="M23 43h18"/>',
  };
  const brows = {
    0: '<path d="M18 18q5-5 10-1M36 17q5-4 10 1"/>',
    1: '<path d="M18 22q5-6 10-3M36 19q5-3 10 3"/>',
    2: '<path d="M18 21l10 2M36 23l10-3"/>',
    4: '<path d="M18 22q6-2 10-6M36 16q4 4 10 6"/>',
    5: '<path d="M18 18l10 5M36 23l10-5"/>',
  };
  const eyes = idx === 3
    ? '<path d="M20 28q4-4 8 0M36 28q4-4 8 0"/>'
    : `<circle cx="24" cy="28" r="${idx === 0 || idx === 1 ? 3.6 : 3}" fill="#1a1a1f" stroke="none"/><circle cx="40" cy="28" r="${idx === 0 || idx === 1 ? 3.6 : 3}" fill="#1a1a1f" stroke="none"/>`;
  return `<svg viewBox="0 0 64 64" width="${size}" height="${size}" style="stroke:#1a1a1f;stroke-width:3">
    <circle cx="32" cy="32" r="29" fill="${c}" stroke="none"/>
    ${brows[idx] || ""}${eyes}${mouths[idx] ?? ""}</svg>`;
}

function barsHTML(probs, compact = false) {
  const top = probs.some(p => p > 0) ? argmax(probs) : -1;
  return EMOTIONS.map((e, i) => `
    <div class="bar${i === top ? " top" : ""}">
      <span class="name">${e}</span>
      <div class="track"><div class="fill" style="width:${(probs[i] * 100).toFixed(1)}%;background:${EMO_COLORS[i]}"></div></div>
      <span class="val">${(probs[i] * 100).toFixed(compact ? 0 : 1)}%</span>
    </div>`).join("");
}

/* ================= State & settings ================= */
const state = {
  info: null,
  model: "efficientnet",
  weights: "current",
  compare: false,
  detect: "auto",
  pad: 5,
  noise: 0,
  denoise: false,
  smooth: 60,
};

function params(extra = {}) {
  const p = new URLSearchParams({
    models: extra.models || state.model,
    weights: state.weights,
    detect: state.detect,
    pad: (state.pad / 100).toString(),
    noise: String(state.noise),
    denoise: state.denoise ? "1" : "0",
    ...extra,
  });
  return p.toString();
}

async function predict(blob, extra) {
  const res = await fetch("/api/predict?" + params(extra), { method: "POST", body: blob });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || "Prediction failed");
  return data;
}

function setRangeFill(input) {
  const p = ((input.value - input.min) / (input.max - input.min)) * 100;
  input.style.setProperty("--p", p + "%");
}

function bindSettings() {
  $$("#modelSeg button").forEach(b => b.addEventListener("click", () => {
    if (b.disabled) return;
    $$("#modelSeg button").forEach(x => x.classList.toggle("on", x === b));
    state.model = b.dataset.v;
    onSettingsChanged();
  }));
  $$("#detectSeg button").forEach(b => b.addEventListener("click", () => {
    $$("#detectSeg button").forEach(x => x.classList.toggle("on", x === b));
    state.detect = b.dataset.v;
    onSettingsChanged();
  }));
  $("#weightsSel").addEventListener("change", e => {
    state.weights = e.target.value;
    refreshModelAvailability();
    onSettingsChanged();
  });
  $("#compareAll").addEventListener("change", e => { state.compare = e.target.checked; onSettingsChanged(); });
  $("#denoise").addEventListener("change", e => { state.denoise = e.target.checked; onSettingsChanged(); });

  const ranges = [
    ["#pad", "pad", v => v + "%"],
    ["#noise", "noise", v => (v == 0 ? "off" : "σ = " + v)],
    ["#smooth", "smooth", v => v + "%"],
  ];
  for (const [sel, key, fmt] of ranges) {
    const input = $(sel);
    const out = $(sel + "Val");
    const update = commit => {
      state[key] = Number(input.value);
      out.textContent = fmt(input.value);
      setRangeFill(input);
      if (commit && key !== "smooth") onSettingsChanged();
    };
    input.addEventListener("input", () => update(false));
    input.addEventListener("change", () => update(true));
    update(false);
  }

  $("#legend").innerHTML = EMOTIONS.map((e, i) => `<div><i style="background:${EMO_COLORS[i]}"></i>${e}</div>`).join("");
}

let settingsTimer;
function onSettingsChanged() {
  clearTimeout(settingsTimer);
  settingsTimer = setTimeout(() => { if (Img.blob) Img.run(); Gan.onSettingsChanged(); }, 120);
}

function refreshModelAvailability() {
  if (!state.info) return;
  const avail = state.info.available[state.weights] || [];
  $$("#modelSeg button").forEach(b => {
    b.disabled = !avail.includes(b.dataset.v);
    b.title = b.disabled ? "Weights file not found for this set" : "";
  });
  if (!avail.includes(state.model) && avail.length) {
    $(`#modelSeg button[data-v="${avail[0]}"]`).click();
  }
}

/* ================= Tabs & theme ================= */
function bindTabs() {
  $$(".tab").forEach(t => t.addEventListener("click", () => {
    $$(".tab").forEach(x => x.classList.toggle("active", x === t));
    $$(".view").forEach(v => v.classList.toggle("active", v.id === "view-" + t.dataset.tab));
    $("#compareRow").classList.toggle("disabled", t.dataset.tab === "live" || t.dataset.tab === "gan");
    if (t.dataset.tab === "bench") Bench.render();
    if (t.dataset.tab === "gan") Gan.start();
    if (t.dataset.tab === "live") Live.resizeTimeline();
    history.replaceState(null, "", "#" + t.dataset.tab);
  }));
  const initial = location.hash.slice(1);
  if (initial) $(`.tab[data-tab="${initial}"]`)?.click();

  const saved = (() => { try { return localStorage.getItem("fn-theme"); } catch { return null; } })();
  if (saved) document.documentElement.dataset.theme = saved;
  $("#themeBtn").addEventListener("click", () => {
    const next = document.documentElement.dataset.theme === "light" ? "dark" : "light";
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem("fn-theme", next); } catch {}
    Bench.rendered = false;
    if ($("#view-bench").classList.contains("active")) Bench.render();
  });
}

/* ================= Live ================= */
const Live = {
  video: null, overlay: null, ctx: null,
  running: false, paused: false, stream: null, isFile: false,
  tracks: [], nextId: 1, frameW: 0, frameH: 0,
  history: [], // {t, probs}
  fpsTimes: [],
  grab: document.createElement("canvas"),

  init() {
    this.video = $("#video");
    this.overlay = $("#overlay");
    this.ctx = this.overlay.getContext("2d");
    this.tl = $("#timeline");
    this.tlCtx = this.tl.getContext("2d");
    $("#liveBars").innerHTML = barsHTML(new Array(7).fill(0));
    $("#domFace").innerHTML = faceSVG(-1);

    $("#startCam").addEventListener("click", () => this.startCamera());
    $("#videoFile").addEventListener("change", e => e.target.files[0] && this.startFile(e.target.files[0]));
    $("#stopCam").addEventListener("click", () => this.stop());
    $("#pauseBtn").addEventListener("click", () => this.togglePause());
    $("#snapBtn").addEventListener("click", () => this.snapshot());
    $("#clearTimeline").addEventListener("click", () => { this.history = []; this.drawTimeline(); this.updateSummary(); });
    $("#mirror").addEventListener("change", () => this.applyMirror());
    $("#camSel").addEventListener("change", () => this.running && !this.isFile && this.startCamera($("#camSel").value));
    window.addEventListener("resize", () => this.resizeTimeline());
    this.resizeTimeline();
    this.listCameras();
    requestAnimationFrame(() => this.render());
  },

  async listCameras() {
    if (!navigator.mediaDevices?.enumerateDevices) return;
    const devices = (await navigator.mediaDevices.enumerateDevices()).filter(d => d.kind === "videoinput");
    const sel = $("#camSel");
    const current = sel.value;
    sel.innerHTML = devices.map((d, i) => `<option value="${esc(d.deviceId)}">${esc(d.label || "Camera " + (i + 1))}</option>`).join("");
    if (current) sel.value = current;
    sel.hidden = devices.length < 2;
  },

  async startCamera(deviceId) {
    if (!navigator.mediaDevices?.getUserMedia) {
      toast("Camera access needs http://localhost or HTTPS", true);
      return;
    }
    try {
      this.stopStream();
      this.stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, ...(deviceId ? { deviceId: { exact: deviceId } } : { facingMode: "user" }) },
        audio: false,
      });
      this.video.srcObject = this.stream;
      this.video.removeAttribute("src");
      this.isFile = false;
      $("#mirror").checked = true;
      await this.video.play();
      this.listCameras();
      this.begin();
    } catch (err) {
      toast("Could not open the camera: " + err.message, true);
    }
  },

  async startFile(file) {
    this.stopStream();
    this.video.srcObject = null;
    this.video.src = URL.createObjectURL(file);
    this.video.loop = true;
    this.isFile = true;
    $("#mirror").checked = false;
    try { await this.video.play(); this.begin(); } catch (err) { toast("Could not play the video: " + err.message, true); }
  },

  begin() {
    $("#liveEmpty").hidden = true;
    $("#hud").hidden = false;
    ["#stopCam", "#pauseBtn", "#snapBtn"].forEach(s => ($(s).disabled = false));
    this.paused = false;
    this.setPauseLabel();
    this.applyMirror();
    this.tracks = [];
    if (!this.running) { this.running = true; this.loop(); }
  },

  stopStream() {
    this.stream?.getTracks().forEach(t => t.stop());
    this.stream = null;
  },

  stop() {
    this.running = false;
    this.stopStream();
    this.video.pause();
    this.video.srcObject = null;
    this.video.removeAttribute("src");
    this.video.load();
    this.tracks = [];
    $("#liveEmpty").hidden = false;
    $("#hud").hidden = true;
    ["#stopCam", "#pauseBtn", "#snapBtn"].forEach(s => ($(s).disabled = true));
    $("#videoFile").value = "";
  },

  togglePause() {
    this.paused = !this.paused;
    if (this.isFile) this.paused ? this.video.pause() : this.video.play();
    this.setPauseLabel();
  },

  setPauseLabel() {
    $("#pauseBtn span").textContent = this.paused ? "Resume" : "Pause";
    $("#pauseBtn svg").innerHTML = this.paused ? '<path d="M7 5l12 7-12 7z"/>' : '<path d="M8 5v14M16 5v14"/>';
  },

  applyMirror() {
    this.video.classList.toggle("mirror", $("#mirror").checked);
  },

  async loop() {
    while (this.running) {
      const v = this.video;
      if (this.paused || v.readyState < 2 || !v.videoWidth) { await sleep(80); continue; }
      const scale = Math.min(1, 640 / v.videoWidth);
      const w = Math.round(v.videoWidth * scale), h = Math.round(v.videoHeight * scale);
      this.grab.width = w; this.grab.height = h;
      this.grab.getContext("2d").drawImage(v, 0, 0, w, h);
      const blob = await new Promise(r => this.grab.toBlob(r, "image/jpeg", 0.85));
      const t0 = performance.now();
      try {
        const res = await predict(blob, { models: state.model, detect: state.detect === "off" ? "off" : "on", max_faces: "4" });
        if (!this.running) break;
        this.frameW = res.width; this.frameH = res.height;
        this.ingest(res);
        const now = performance.now();
        $("#latency").textContent = Math.round(now - t0);
        this.fpsTimes.push(now);
        this.fpsTimes = this.fpsTimes.filter(t => now - t < 1000);
        $("#fps").textContent = this.fpsTimes.length;
      } catch (err) {
        toast(err.message, true);
        await sleep(1000);
      }
    }
  },

  ingest(res) {
    const now = performance.now();
    const alpha = state.smooth / 100;
    const faces = res.faces.map(f => ({ box: f.box, probs: f.results[state.model]?.probs || Object.values(f.results)[0].probs }));
    const used = new Set();
    for (const f of faces) {
      let best = null, bestIoU = 0.15;
      for (const t of this.tracks) {
        if (used.has(t)) continue;
        const iou = IoU(t.box, f.box);
        if (iou > bestIoU) { best = t; bestIoU = iou; }
      }
      if (best) {
        used.add(best);
        best.box = f.box;
        best.probs = best.probs.map((p, i) => alpha * p + (1 - alpha) * f.probs[i]);
        best.seen = now;
      } else {
        const t = { id: this.nextId++, box: f.box, draw: [...f.box], probs: f.probs, seen: now };
        this.tracks.push(t);
        used.add(t);
      }
    }
    this.tracks = this.tracks.filter(t => now - t.seen < 700);
    $("#faceCount").textContent = faces.length;

    const primary = this.primary();
    if (primary) {
      this.updatePanel(primary.probs);
      this.history.push({ t: Date.now(), probs: primary.probs });
    } else {
      this.updatePanel(null);
    }
    const cutoff = Date.now() - 60000;
    while (this.history.length && this.history[0].t < cutoff) this.history.shift();
    this.drawTimeline();
    this.updateSummary();
  },

  primary() {
    return this.tracks.reduce((a, t) => (!a || t.box[2] * t.box[3] > a.box[2] * a.box[3] ? t : a), null);
  },

  updatePanel(probs) {
    const dom = $("#dominant");
    if (!probs) {
      $("#domName").textContent = "—";
      $("#domConf").textContent = "No face detected";
      $("#domFace").innerHTML = faceSVG(-1);
      dom.style.setProperty("--emo", "transparent");
      $("#liveBars").innerHTML = barsHTML(new Array(7).fill(0));
      return;
    }
    const top = argmax(probs);
    if (this._lastTop !== top) {
      $("#domFace").innerHTML = faceSVG(top);
      this._lastTop = top;
    }
    $("#domName").textContent = EMOTIONS[top];
    $("#domConf").textContent = pct(probs[top]) + " confidence";
    dom.style.setProperty("--emo", EMO_COLORS[top]);
    // Update widths in place so the CSS transition animates
    const bars = $$("#liveBars .bar");
    bars.forEach((b, i) => {
      b.classList.toggle("top", i === top);
      $(".fill", b).style.width = (probs[i] * 100).toFixed(1) + "%";
      $(".val", b).textContent = (probs[i] * 100).toFixed(1) + "%";
    });
  },

  render() {
    requestAnimationFrame(() => this.render());
    const c = this.overlay, ctx = this.ctx;
    const dpr = window.devicePixelRatio || 1;
    const cw = c.clientWidth, ch = c.clientHeight;
    if (c.width !== Math.round(cw * dpr) || c.height !== Math.round(ch * dpr)) {
      c.width = Math.round(cw * dpr); c.height = Math.round(ch * dpr);
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cw, ch);
    if (!this.running || !this.frameW) return;
    this.drawOverlay(ctx, cw, ch, $("#mirror").checked, true);
  },

  drawOverlay(ctx, cw, ch, mirror, animate) {
    const s = Math.min(cw / this.frameW, ch / this.frameH);
    const ox = (cw - this.frameW * s) / 2, oy = (ch - this.frameH * s) / 2;
    const primary = this.primary();
    for (const t of this.tracks) {
      if (animate) t.draw = t.draw.map((v, i) => v + (t.box[i] - v) * 0.35);
      let [x, y, w, h] = animate ? t.draw : t.box;
      if (mirror) x = this.frameW - x - w;
      const X = ox + x * s, Y = oy + y * s, W = w * s, H = h * s;
      const top = argmax(t.probs);
      const color = EMO_COLORS[top];
      const r = Math.min(14, W * 0.12);

      ctx.lineWidth = t === primary ? 3 : 2;
      ctx.strokeStyle = color;
      ctx.shadowColor = color; ctx.shadowBlur = 12;
      // Corner brackets
      const L = Math.min(W, H) * 0.22;
      ctx.beginPath();
      ctx.moveTo(X, Y + L); ctx.lineTo(X, Y + r); ctx.quadraticCurveTo(X, Y, X + r, Y); ctx.lineTo(X + L, Y);
      ctx.moveTo(X + W - L, Y); ctx.lineTo(X + W - r, Y); ctx.quadraticCurveTo(X + W, Y, X + W, Y + r); ctx.lineTo(X + W, Y + L);
      ctx.moveTo(X + W, Y + H - L); ctx.lineTo(X + W, Y + H - r); ctx.quadraticCurveTo(X + W, Y + H, X + W - r, Y + H); ctx.lineTo(X + W - L, Y + H);
      ctx.moveTo(X + L, Y + H); ctx.lineTo(X + r, Y + H); ctx.quadraticCurveTo(X, Y + H, X, Y + H - r); ctx.lineTo(X, Y + H - L);
      ctx.stroke();
      ctx.shadowBlur = 0;
      ctx.lineWidth = 1; ctx.globalAlpha = 0.35;
      ctx.strokeRect(X, Y, W, H);
      ctx.globalAlpha = 1;

      // Label pill
      const label = `${EMOTIONS[top]}  ${Math.round(t.probs[top] * 100)}%`;
      ctx.font = "600 13px -apple-system, BlinkMacSystemFont, sans-serif";
      const tw = ctx.measureText(label).width;
      const pw = tw + 22, phh = 24;
      let py = Y - phh - 8;
      if (py < 4) py = Y + H + 8;
      ctx.fillStyle = color;
      roundRect(ctx, X, py, pw, phh, 7); ctx.fill();
      ctx.fillStyle = "rgba(10,10,14,.9)";
      ctx.fillText(label, X + 11, py + 16.5);

      // Mini probability strip under the box
      const sy = Math.min(Y + H + 6, ch - 8);
      if (py > Y || sy + 6 > ch) continue;
      let sx = X;
      for (let i = 0; i < 7; i++) {
        const sw = W * t.probs[i];
        ctx.fillStyle = EMO_COLORS[i];
        ctx.fillRect(sx, sy, sw, 4);
        sx += sw;
      }
    }
  },

  resizeTimeline() {
    const dpr = window.devicePixelRatio || 1;
    const w = this.tl.clientWidth;
    if (!w) return;
    this.tl.width = w * dpr; this.tl.height = 140 * dpr;
    this.drawTimeline();
  },

  drawTimeline() {
    const ctx = this.tlCtx, dpr = window.devicePixelRatio || 1;
    const W = this.tl.width / dpr, H = 140;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    const css = getComputedStyle(document.documentElement);
    ctx.strokeStyle = css.getPropertyValue("--line");
    ctx.fillStyle = css.getPropertyValue("--text-3");
    ctx.font = "10.5px ui-monospace, Menlo, monospace";
    ctx.lineWidth = 1;
    const padB = 18;
    for (let s = 0; s <= 60; s += 10) {
      const x = W - (s / 60) * W;
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H - padB); ctx.stroke();
      ctx.textAlign = s === 0 ? "right" : s === 60 ? "left" : "center";
      ctx.fillText(s === 0 ? "now" : `-${s}s`, x, H - 4);
    }
    const hist = this.history;
    if (hist.length < 2) {
      ctx.textAlign = "center";
      ctx.fillText("Start the camera to record emotions over time", W / 2, (H - padB) / 2);
      return;
    }
    const now = Date.now();
    const xs = hist.map(p => W - ((now - p.t) / 60000) * W);
    const ph = H - padB;
    const base = new Array(hist.length).fill(0);
    for (let i = 0; i < 7; i++) {
      ctx.beginPath();
      hist.forEach((p, j) => { const y = ph - (base[j] + p.probs[i]) * ph; j ? ctx.lineTo(xs[j], y) : ctx.moveTo(xs[j], y); });
      for (let j = hist.length - 1; j >= 0; j--) ctx.lineTo(xs[j], ph - base[j] * ph);
      ctx.closePath();
      ctx.fillStyle = EMO_COLORS[i];
      ctx.globalAlpha = 0.85;
      ctx.fill();
      hist.forEach((p, j) => (base[j] += p.probs[i]));
    }
    ctx.globalAlpha = 1;
  },

  updateSummary() {
    const counts = new Array(7).fill(0);
    this.history.forEach(p => counts[argmax(p.probs)]++);
    const n = this.history.length;
    $("#summary").innerHTML = n ? counts
      .map((c, i) => [c, i]).filter(([c]) => c).sort((a, b) => b[0] - a[0])
      .map(([c, i]) => `<span class="pill"><i style="background:${EMO_COLORS[i]}"></i>${EMOTIONS[i]} <b>${Math.round((c / n) * 100)}%</b></span>`)
      .join("") : "";
  },

  snapshot() {
    const v = this.video;
    if (!v.videoWidth) return;
    const c = document.createElement("canvas");
    c.width = v.videoWidth; c.height = v.videoHeight;
    const ctx = c.getContext("2d");
    const mirror = $("#mirror").checked;
    if (mirror) { ctx.translate(c.width, 0); ctx.scale(-1, 1); }
    ctx.drawImage(v, 0, 0);
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    if (this.frameW) this.drawOverlay(ctx, c.width, c.height, mirror, false);
    const url = c.toDataURL("image/jpeg", 0.9);
    const p = this.primary();
    const label = p ? `${EMOTIONS[argmax(p.probs)]} · ${pct(Math.max(...p.probs), 0)}` : "No face";
    const el = document.createElement("div");
    el.className = "snap";
    el.innerHTML = `<img src="${url}" alt=""><div><span>${label}</span><a href="${url}" download="facialnet-${Date.now()}.jpg">Save</a></div>`;
    $("#snapshots").prepend(el);
  },
};

function IoU(a, b) {
  const x0 = Math.max(a[0], b[0]), y0 = Math.max(a[1], b[1]);
  const x1 = Math.min(a[0] + a[2], b[0] + b[2]), y1 = Math.min(a[1] + a[3], b[1] + b[3]);
  const inter = Math.max(0, x1 - x0) * Math.max(0, y1 - y0);
  return inter / (a[2] * a[3] + b[2] * b[3] - inter || 1);
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y); ctx.arcTo(x + w, y, x + w, y + h, r); ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r); ctx.arcTo(x, y, x + w, y, r); ctx.closePath();
}

/* ================= Image ================= */
const Img = {
  blob: null, groundTruth: null, name: "", seq: 0,

  init() {
    const input = $("#fileInput");
    const browse = () => input.click();
    $("#browseBtn").addEventListener("click", e => { e.stopPropagation(); browse(); });
    $("#browseBtn2").addEventListener("click", browse);
    $("#sampleBtn").addEventListener("click", e => { e.stopPropagation(); this.sample(); });
    $("#sampleBtn2").addEventListener("click", () => this.sample());
    input.addEventListener("change", () => { if (input.files[0]) this.load(input.files[0], input.files[0].name); input.value = ""; });

    const drop = $("#drop");
    ["dragenter", "dragover"].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.add("drag"); }));
    ["dragleave", "drop"].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.remove("drag"); }));
    drop.addEventListener("drop", e => {
      const f = [...e.dataTransfer.files].find(f => f.type.startsWith("image/"));
      if (f) this.load(f, f.name);
    });
    document.addEventListener("paste", e => {
      const item = [...(e.clipboardData?.items || [])].find(i => i.type.startsWith("image/"));
      if (!item) return;
      $('.tab[data-tab="image"]').click();
      this.load(item.getAsFile(), "pasted image");
    });

    $("#sampleChips").innerHTML = EMOTIONS.map((e, i) =>
      `<button data-i="${i}"><i style="background:${EMO_COLORS[i]}"></i>${e}</button>`).join("");
    $$("#sampleChips button").forEach(b => b.addEventListener("click", () => this.sample(Number(b.dataset.i))));
    window.addEventListener("resize", () => this.lastResult && this.drawBoxes(this.lastResult));
  },

  async sample(label) {
    try {
      const q = label === undefined ? "" : "?label=" + label;
      const res = await fetch("/api/sample" + q);
      const data = await res.json();
      if (!res.ok) throw new Error(data.error);
      const blob = await (await fetch(data.image)).blob();
      this.load(blob, data.name, data.label);
    } catch (err) {
      toast(err.message, true);
    }
  },

  load(blob, name, groundTruth = null, truthName = "ground truth") {
    this.blob = blob;
    this.name = name;
    this.groundTruth = groundTruth;
    this.truthName = truthName;
    this.lastResult = null;
    const img = $("#imgEl");
    if (img.src.startsWith("blob:")) URL.revokeObjectURL(img.src);
    img.src = URL.createObjectURL(blob);
    img.onload = () => {
      img.classList.toggle("tiny", img.naturalWidth < 300);
      $("#imgBoxes").innerHTML = "";
      this.run();
    };
    $("#dropEmpty").hidden = true;
    $("#imgStage").hidden = false;
  },

  async run() {
    if (!this.blob) return;
    const seq = ++this.seq;
    $("#imgLoading").hidden = false;
    const models = state.compare ? MODEL_ORDER.filter(m => (state.info?.available[state.weights] || []).includes(m)).join(",") : state.model;
    try {
      const res = await predict(this.blob, { models, crops: "1", max_faces: "8" });
      if (seq !== this.seq) return;
      this.lastResult = res;
      this.drawBoxes(res);
      this.renderResults(res);
    } catch (err) {
      if (seq === this.seq) toast(err.message, true);
    } finally {
      if (seq === this.seq) $("#imgLoading").hidden = true;
    }
  },

  primaryResult(face) {
    return face.results[state.model] || Object.values(face.results)[0];
  },

  drawBoxes(res) {
    const img = $("#imgEl"), svg = $("#imgBoxes"), stage = $("#imgStage");
    const ir = img.getBoundingClientRect(), sr = stage.getBoundingClientRect();
    Object.assign(svg.style, { left: ir.left - sr.left + "px", top: ir.top - sr.top + "px", width: ir.width + "px", height: ir.height + "px" });
    svg.setAttribute("viewBox", `0 0 ${res.width} ${res.height}`);
    svg.setAttribute("preserveAspectRatio", "none");
    const k = res.width / ir.width; // image px per screen px
    svg.innerHTML = res.mode === "whole" ? "" : res.faces.map((f, n) => {
      const r = this.primaryResult(f);
      const [x, y, w, h] = f.box;
      const c = EMO_COLORS[r.top];
      const fs = 12 * k, ph = 20 * k;
      const label = `${n + 1} · ${r.label} ${Math.round(r.confidence * 100)}%`;
      const pw = (label.length * 6.6 + 14) * k;
      const py = y - ph - 4 * k < 0 ? y + h + 4 * k : y - ph - 4 * k;
      return `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${6 * k}" fill="none" stroke="${c}" stroke-width="${2.5 * k}"/>
        <rect x="${x}" y="${py}" width="${pw}" height="${ph}" rx="${5 * k}" fill="${c}" stroke="none"/>
        <text x="${x + 7 * k}" y="${py + ph * 0.7}" font-size="${fs}" font-weight="600" fill="#111" stroke="none" font-family="-apple-system,sans-serif">${esc(label)}</text>`;
    }).join("");
  },

  renderResults(res) {
    const t = res.timing;
    let meta = `<span>${esc(this.name)}</span><span>${res.width}×${res.height}</span>`;
    meta += `<span>${res.mode === "whole" ? "whole image classified" : res.faces.length + " face" + (res.faces.length === 1 ? "" : "s") + " detected"}</span>`;
    meta += `<span>${Math.round(t.detect_ms + t.infer_ms)} ms</span>`;
    if (this.groundTruth !== null && res.faces[0]) {
      const pred = this.primaryResult(res.faces[0]).top;
      const ok = pred === this.groundTruth;
      meta += `<span class="${ok ? "gt-ok" : "gt-bad"}">${this.truthName}: ${EMOTIONS[this.groundTruth]} ${ok ? "✓" : "✗"}</span>`;
    }
    $("#imgMeta").innerHTML = meta;

    const col = $("#faceResults");
    if (!res.faces.length) {
      col.innerHTML = `<div class="panel placeholder"><p class="muted">No face found. Switch face detection to <b>Auto</b> or <b>Whole image</b> to classify the full image.</p></div>`;
      return;
    }
    col.innerHTML = res.faces.map((f, n) => this.faceCard(f, n, res)).join("");
  },

  faceCard(face, n, res) {
    const r = this.primaryResult(face);
    const s = face.stages;
    const arrow = '<span class="arrow"><svg viewBox="0 0 24 24"><path d="M5 12h14M13 6l6 6-6 6"/></svg></span>';
    let pipeline = "";
    if (s.noisy || s.denoised) {
      const steps = [`<figure><img src="${s.input}" alt=""><figcaption>Input</figcaption></figure>`];
      if (s.noisy) steps.push(`<figure><img src="${s.noisy}" alt=""><figcaption>+ noise σ=${state.noise}</figcaption></figure>`);
      if (s.denoised) steps.push(`<figure><img src="${s.denoised}" alt=""><figcaption>DnCNN</figcaption></figure>`);
      pipeline = `<div class="pipeline">${steps.join(arrow)}</div>`;
    }
    const thumb = s.denoised || s.noisy || s.input;
    const where = res.mode === "whole" ? "whole image" : `face ${n + 1} of ${res.faces.length}`;

    let body;
    const archs = Object.keys(face.results);
    if (archs.length > 1) {
      const tops = archs.map(a => face.results[a].top);
      const agree = tops.every(x => x === tops[0]);
      body = `<div class="compare">${archs.map(a => {
        const m = face.results[a];
        return `<div class="compare-model">
          <div class="cm-head"><b style="color:${MODEL_COLORS[a]}">${MODEL_LABELS[a]}</b><span>${m.label} · ${pct(m.confidence)}</span></div>
          <div class="stack">${m.probs.map((p, i) => `<div style="width:${p * 100}%;background:${EMO_COLORS[i]}" title="${EMOTIONS[i]} ${pct(p)}">${p > 0.12 ? `<span>${EMOTIONS[i]}</span>` : ""}</div>`).join("")}</div>
        </div>`;
      }).join("")}
      <div class="agree">${agree ? `All ${archs.length} models agree: <b>${EMOTIONS[tops[0]]}</b>` : `Models disagree: ${archs.map(a => `${MODEL_LABELS[a]} → <b>${face.results[a].label}</b>`).join(", ")}`}</div></div>`;
    } else {
      body = `<div class="bars">${barsHTML(r.probs)}</div>`;
    }

    return `<div class="panel face-card" style="--emo:${EMO_COLORS[r.top]}">
      <div class="face-head">
        <img class="face-thumb" src="${thumb}" alt="">
        <div style="flex:1">
          <div class="face-index">${where}</div>
          <div class="face-title">${r.label}</div>
          <div class="face-sub">${pct(r.confidence)} · ${MODEL_LABELS[archs.length > 1 ? state.model : archs[0]] || ""}</div>
        </div>
        <div class="emo-face">${faceSVG(r.top, 48)}</div>
      </div>
      ${pipeline}${body}
    </div>`;
  },
};

/* ================= GAN ================= */
const Gan = {
  started: false, label: "all", count: 8, seed: 0, morphSeed: 0, identitySeed: 0, morphIndex: 0,
  seq: {}, results: {}, settingsKey: "",

  init() {
    $("#ganChips").innerHTML = `<button data-v="all" class="on">All emotions</button>` + EMOTIONS.map((e, i) =>
      `<button data-v="${i}"><i style="background:${EMO_COLORS[i]}"></i>${e}</button>`).join("");
    $$("#ganChips button").forEach(b => b.addEventListener("click", () => {
      $$("#ganChips button").forEach(x => x.classList.toggle("on", x === b));
      this.label = b.dataset.v;
      this.runGrid();
    }));
    $$("#ganCount button").forEach(b => b.addEventListener("click", () => {
      $$("#ganCount button").forEach(x => x.classList.toggle("on", x === b));
      this.count = Number(b.dataset.v);
      this.runGrid();
    }));
    $("#ganSeed").addEventListener("change", e => {
      const v = Math.max(0, Math.min(999999, Math.floor(Number(e.target.value) || 0)));
      e.target.value = v;
      this.seed = v;
      this.runGrid();
    });
    $("#ganNew").addEventListener("click", () => { this.seed = randomSeed(); this.runGrid(); });

    const opts = EMOTIONS.map((e, i) => `<option value="${i}">${e}</option>`).join("");
    $("#morphFrom").innerHTML = opts;
    $("#morphTo").innerHTML = opts;
    $("#morphFrom").value = "6";
    $("#morphTo").value = "3";
    ["#morphFrom", "#morphTo"].forEach(s => $(s).addEventListener("change", () => this.runMorph()));
    $("#morphT").addEventListener("input", e => this.showMorph(Number(e.target.value)));
    $("#morphShuffle").addEventListener("click", () => { this.morphSeed = randomSeed(); this.runMorph(); });
    $("#identityShuffle").addEventListener("click", () => { this.identitySeed = randomSeed(); this.runIdentity(); });
  },

  // Called when the tab opens: the first visit draws new random faces
  start() {
    this.renderStats();
    if (this.started) return;
    this.started = true;
    if (state.info && !state.info.gan) { this.showMissing(); return; }
    this.seed = randomSeed();
    this.morphSeed = randomSeed();
    this.identitySeed = randomSeed();
    this.runAll();
  },

  showMissing() {
    $("#ganMissing").hidden = false;
    $$("#view-gan > :not(#ganMissing)").forEach(el => (el.hidden = true));
  },

  // Re-check the same faces when the classifier in the sidebar changes
  onSettingsChanged() {
    const key = state.model + "/" + state.weights;
    if (key === this.settingsKey) return;
    this.settingsKey = key;
    if (this.started && $("#ganMissing").hidden) this.runAll();
  },

  runAll() {
    this.settingsKey = state.model + "/" + state.weights;
    this.runGrid();
    this.runIdentity();
    this.runMorph();
  },

  async request(kind, params, busyEl) {
    const seq = (this.seq[kind] || 0) + 1;
    this.seq[kind] = seq;
    busyEl.classList.add("busy");
    try {
      const q = new URLSearchParams({ model: state.model, weights: state.weights, ...params });
      const res = await fetch("/api/generate?" + q);
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Generation failed");
      return seq === this.seq[kind] ? data : null;
    } catch (err) {
      if (seq === this.seq[kind]) toast(err.message, true);
      return null;
    } finally {
      if (seq === this.seq[kind]) busyEl.classList.remove("busy");
    }
  },

  async runGrid() {
    $("#ganSeed").value = this.seed;
    const data = await this.request("grid", { mode: "label", label: this.label, n: this.count, seed: this.seed }, $("#ganFaces"));
    if (!data) return;
    this.results.grid = data;
    $("#ganFaces").innerHTML = data.faces.map((f, i) => this.card(f, i, "grid")).join("");
    this.bindCards($("#ganFaces"), "grid");
    const hits = data.faces.filter(f => f.top === f.target).length;
    const t = data.timing;
    $("#ganSummary").innerHTML = `${MODEL_LABELS[data.model]} recognises the requested emotion in <b>${hits} of ${data.faces.length}</b> faces · ` +
      `generated in <b>${Math.round(t.generate_ms)} ms</b>, classified in <b>${Math.round(t.classify_ms)} ms</b> · click a face to analyse it in the Image tab`;
  },

  async runIdentity() {
    const data = await this.request("identity", { mode: "emotions", seed: this.identitySeed }, $("#identityRow"));
    if (!data) return;
    this.results.identity = data;
    $("#identityRow").innerHTML = data.faces.map((f, i) => this.card(f, i, "identity")).join("");
    this.bindCards($("#identityRow"), "identity");
  },

  async runMorph() {
    const steps = Number($("#morphT").max) + 1;
    const params = { mode: "morph", from: $("#morphFrom").value, to: $("#morphTo").value, steps, seed: this.morphSeed };
    const data = await this.request("morph", params, $("#morphMain"));
    if (!data) return;
    this.results.morph = data;
    $("#morphStrip").innerHTML = data.faces.map((f, i) => `<img src="${f.image}" data-i="${i}" alt="Step ${i + 1}">`).join("");
    $$("#morphStrip img").forEach(img => img.addEventListener("click", () => {
      $("#morphT").value = img.dataset.i;
      this.showMorph(Number(img.dataset.i));
    }));
    this.showMorph(Number($("#morphT").value));
  },

  showMorph(i) {
    const data = this.results.morph;
    if (!data) return;
    const f = data.faces[i];
    const a = Number($("#morphFrom").value), b = Number($("#morphTo").value);
    $("#morphImg").src = f.image;
    $("#morphMain").style.setProperty("--emo", EMO_COLORS[f.top]);
    $("#morphBars").innerHTML = barsHTML(f.probs, true);
    const pb = Math.round(f.mix[b] * 100);
    $("#morphVal").textContent = a === b ? EMOTIONS[a] : `${EMOTIONS[a]} ${100 - pb}% · ${EMOTIONS[b]} ${pb}%`;
    $$("#morphStrip img").forEach((img, j) => img.classList.toggle("on", j === i));
    setRangeFill($("#morphT"));
  },

  card(f, i, kind) {
    const ok = f.top === f.target;
    const verdict = ok ? `✓ ${pct(f.confidence, 0)}` : `✗ ${f.label} ${pct(f.confidence, 0)}`;
    const tip = `Requested ${EMOTIONS[f.target]} · ${MODEL_LABELS[this.results[kind]?.model || state.model]} sees ${f.label} (${pct(f.confidence)})`;
    return `<button class="gan-face" data-i="${i}" style="--emo:${EMO_COLORS[f.target]}" title="${esc(tip)}">
      <img src="${f.image}" alt="Generated ${EMOTIONS[f.target]} face">
      <div class="gf-cap"><span class="req"><i></i>${EMOTIONS[f.target]}</span><span class="got ${ok ? "ok" : "bad"}">${verdict}</span></div>
    </button>`;
  },

  bindCards(root, kind) {
    $$(".gan-face", root).forEach(el => el.addEventListener("click", () => this.analyse(this.results[kind], Number(el.dataset.i))));
  },

  // Open a generated face in the Image tab for the full per-model analysis
  async analyse(data, i) {
    const f = data.faces[i];
    const blob = await (await fetch(f.image)).blob();
    $('.tab[data-tab="image"]').click();
    Img.load(blob, `GAN face · seed ${data.seed} · ${EMOTIONS[f.target]}`, f.target, "requested");
  },

  renderStats() {
    const r = state.info?.gan_results;
    if (!r) {
      $("#ganAgreeChart").innerHTML = '<p class="muted">GAN/results.json not found.</p>';
      $("#ganStats").innerHTML = '<p class="muted">GAN/results.json not found.</p>';
      return;
    }
    const models = r.classifier_agreement.models;
    const names = Object.keys(models);
    const mean = kind => EMOTIONS.map((_, c) => names.reduce((s, n) => s + models[n][kind][c], 0) / names.length);
    const real = mean("real_64px"), gen = mean("generated");
    const avg = a => a.reduce((s, v) => s + v, 0) / a.length;

    const W = 560, rowH = 34, padL = 92, padR = 46, padT = 4;
    const H = padT + EMOTIONS.length * rowH + 22;
    const x = v => padL + v * (W - padL - padR);
    const C = ["#8a90a2", "var(--accent)"];
    let s = `<svg viewBox="0 0 ${W} ${H}">`;
    for (let v = 0; v <= 1.0001; v += 0.2) s += `<line class="grid" x1="${x(v)}" x2="${x(v)}" y1="${padT}" y2="${H - 20}"/><text x="${x(v)}" y="${H - 6}" text-anchor="middle">${Math.round(v * 100)}%</text>`;
    EMOTIONS.forEach((e, i) => {
      const y0 = padT + i * rowH;
      const tip = `<b>${e}</b><br>` + names.map(n => `${n}: real ${pct(models[n].real_64px[i])} · generated ${pct(models[n].generated[i])}`).join("<br>");
      s += `<g data-tip="${esc(tip)}"><rect x="0" y="${y0}" width="${W}" height="${rowH}" fill="transparent" stroke="none"/>`;
      s += `<circle cx="10" cy="${y0 + rowH / 2}" r="4" fill="${EMO_COLORS[i]}" stroke="none"/>`;
      s += `<text class="lbl" x="20" y="${y0 + rowH / 2}" dominant-baseline="middle">${e}</text>`;
      [real[i], gen[i]].forEach((v, j) => {
        const by = y0 + 6 + j * 12;
        s += `<rect x="${x(0)}" y="${by}" width="${Math.max(0, x(v) - x(0))}" height="10" rx="2.5" fill="${C[j]}" stroke="none"/>`;
        s += `<text class="val" x="${x(v) + 5}" y="${by + 8.5}">${Math.round(v * 100)}</text>`;
      });
      s += `</g>`;
    });
    s += "</svg>";
    $("#ganAgreeChart").innerHTML = s + legendHTML([["Real test faces (64×64)", C[0]], ["Generated faces", C[1]]]) +
      `<p class="hint" style="margin-top:8px">Share of faces that the classifiers label with the emotion the generator was asked for (${r.classifier_agreement.per_class_generated} generated faces per emotion). Real faces are downscaled to the GAN's 64×64 size for a fair reference.</p>`;
    attachTips($("#ganAgreeChart"));

    const m = r.model, t = r.training, fd = r.frechet_distance;
    const rows = names.map(n => `<tr><td>${n}</td><td>${pct(avg(models[n].real_64px))}</td><td>${pct(avg(models[n].generated))}</td></tr>`).join("");
    $("#ganStats").innerHTML = `
      <div class="stat-tiles">
        <div><span>Generator</span><b>${(m.generator_parameters / 1e6).toFixed(1)}M</b><small>parameters</small></div>
        <div><span>Training</span><b>${t.iterations.toLocaleString()}</b><small>iterations · ${Math.round(t.minutes)} min</small></div>
        <div><span>Fréchet distance</span><b>${fd.generated_ema_vs_test.toFixed(1)}</b><small>real vs real: ${fd.real_train_vs_test.toFixed(1)}</small></div>
      </div>
      <table class="data"><thead><tr><th>Classifier</th><th>Real 64×64</th><th>Generated</th></tr></thead><tbody>${rows}</tbody></table>
      <p class="hint" style="margin-top:10px">Mean over the seven emotions. Fréchet distance between ${fd.images_per_set.toLocaleString()} generated and real test faces in EfficientNet-B0 feature space (lower is better; without EMA: ${fd.generated_raw_vs_test.toFixed(1)}).
      Conditional DCGAN with emotion-conditional BatchNorm in the generator and a projection discriminator, trained on ${r.train_images.toLocaleString()} RAF-DB faces at ${r.image_size}×${r.image_size}.</p>`;
  },
};

function randomSeed() {
  return Math.floor(Math.random() * 1000000);
}

/* ================= Benchmarks ================= */
const Tip = {
  el: null,
  show(e, html) {
    if (!this.el) { this.el = document.createElement("div"); this.el.className = "tip"; document.body.appendChild(this.el); }
    this.el.innerHTML = html;
    this.el.hidden = false;
    this.el.style.left = e.clientX + 12 + "px";
    this.el.style.top = e.clientY + 12 + "px";
  },
  hide() { if (this.el) this.el.hidden = true; },
};

function attachTips(root) {
  $$("[data-tip]", root).forEach(el => {
    el.addEventListener("mousemove", e => Tip.show(e, el.dataset.tip));
    el.addEventListener("mouseleave", () => Tip.hide());
  });
}

function legendHTML(items) {
  return `<div class="chart-legend">${items.map(([label, color, dashed]) =>
    `<span><i style="background:${dashed ? `repeating-linear-gradient(90deg,${color} 0 3px,transparent 3px 5px)` : color}"></i>${label}</span>`).join("")}</div>`;
}

const Bench = {
  rendered: false,
  render() {
    if (this.rendered) return;
    this.rendered = true;
    this.cards();
    this.perClass();
    this.aug();
    this.noise();
    this.psnr();
    this.balance();
  },

  cards() {
    const best = Math.max(...MODEL_ORDER.map(m => BENCH.models[m].acc));
    $("#modelCards").innerHTML = MODEL_ORDER.map(m => {
      const b = BENCH.models[m];
      const n = state.info?.results?.classifier_accuracy?.[MODEL_LABELS[m]];
      const sigma50 = n ? `${pct(n.noisy_50)} → ${pct(n["fine-tuned_50"])}` : "—";
      return `<div class="panel model-card" style="--c:${MODEL_COLORS[m]}">
        <div class="mc-name"><i></i>${MODEL_LABELS[m]}${b.acc === best ? '<span class="badge">Best</span>' : ""}${m === "efficientnet" ? '<span class="badge">Smallest</span>' : ""}</div>
        <div class="mc-acc">${b.acc.toFixed(2)}%<small>RAF-DB test accuracy</small></div>
        <div class="mc-stats">
          <div><span>Parameters</span><b>${b.params}</b></div>
          <div><span>Weights</span><b>${b.size}</b></div>
          <div><span>Per epoch</span><b>${b.epoch}</b></div>
        </div>
        <div class="muted small">σ = 50 noise: ${sigma50} with DnCNN</div>
      </div>`;
    }).join("");
  },

  perClass() {
    const W = 560, rowH = 46, padL = 92, padR = 46, padT = 6;
    const H = padT + EMOTIONS.length * rowH + 22;
    const x = v => padL + ((v - 40) / 60) * (W - padL - padR);
    let s = `<svg viewBox="0 0 ${W} ${H}">`;
    for (let v = 40; v <= 100; v += 10) s += `<line class="grid" x1="${x(v)}" x2="${x(v)}" y1="${padT}" y2="${H - 20}"/><text x="${x(v)}" y="${H - 6}" text-anchor="middle">${v}%</text>`;
    EMOTIONS.forEach((e, i) => {
      const y0 = padT + i * rowH;
      s += `<circle cx="10" cy="${y0 + rowH / 2 - 2}" r="4" fill="${EMO_COLORS[i]}" stroke="none"/>`;
      s += `<text class="lbl" x="20" y="${y0 + rowH / 2 - 2}" dominant-baseline="middle">${e}</text>`;
      s += `<text x="20" y="${y0 + rowH / 2 + 11}" dominant-baseline="middle" font-size="9.5">n=${BENCH.testCounts[i]}</text>`;
      MODEL_ORDER.forEach((m, j) => {
        const v = BENCH.perClass[m][i];
        const by = y0 + 6 + j * 11;
        const isBest = v === Math.max(...MODEL_ORDER.map(mm => BENCH.perClass[mm][i]));
        s += `<rect class="hit" data-tip="<b>${MODEL_LABELS[m]}</b> · ${e}: ${v}%" x="${x(40)}" y="${by}" width="${x(v) - x(40)}" height="9" rx="2.5" fill="${MODEL_COLORS[m]}" stroke="none"/>`;
        if (isBest) s += `<text class="val" x="${x(v) + 5}" y="${by + 8}">${v}</text>`;
      });
    });
    s += "</svg>";
    $("#perClassChart").innerHTML = s + legendHTML(MODEL_ORDER.map(m => [MODEL_LABELS[m], MODEL_COLORS[m]]));
    attachTips($("#perClassChart"));
  },

  aug() {
    const conds = BENCH.aug.conditions;
    const W = 560, rowH = 50, padL = 92, padR = 20, padT = 6;
    const H = padT + conds.length * rowH + 22;
    const x = v => padL + ((v - 72) / 18) * (W - padL - padR);
    let s = `<svg viewBox="0 0 ${W} ${H}">`;
    for (let v = 72; v <= 90; v += 3) s += `<line class="grid" x1="${x(v)}" x2="${x(v)}" y1="${padT}" y2="${H - 20}"/><text x="${x(v)}" y="${H - 6}" text-anchor="middle">${v}%</text>`;
    conds.forEach((c, i) => {
      const y0 = padT + i * rowH;
      s += `<text class="lbl" x="0" y="${y0 + rowH / 2}" dominant-baseline="middle">${c}</text>`;
      MODEL_ORDER.forEach((m, j) => {
        const [a, b] = BENCH.aug[m][i];
        const y = y0 + 12 + j * 13;
        const col = MODEL_COLORS[m];
        const d = (b - a).toFixed(1);
        s += `<g data-tip="<b>${MODEL_LABELS[m]}</b> · ${c}<br>${a}% → ${b}% (${d >= 0 ? "+" : ""}${d})">
          <line x1="${x(a)}" x2="${x(b)}" y1="${y}" y2="${y}" stroke="${col}" stroke-width="2.5" opacity=".5"/>
          <circle cx="${x(a)}" cy="${y}" r="4" fill="var(--panel)" stroke="${col}" stroke-width="2"/>
          <circle cx="${x(b)}" cy="${y}" r="4.5" fill="${col}" stroke="none"/>
          <rect x="${Math.min(x(a), x(b)) - 6}" y="${y - 6}" width="${Math.abs(x(b) - x(a)) + 12}" height="12" fill="transparent" stroke="none"/></g>`;
      });
    });
    s += "</svg>";
    $("#augChart").innerHTML = s + legendHTML([...MODEL_ORDER.map(m => [MODEL_LABELS[m], MODEL_COLORS[m]])]) +
      `<p class="hint" style="margin-top:8px">Hollow dot: flip + rotation only. Filled dot: with zoom, colour jitter and random erasing.</p>`;
    attachTips($("#augChart"));
  },

  noise() {
    const acc = state.info?.results?.classifier_accuracy;
    const el = $("#noiseChart");
    if (!acc) { el.innerHTML = '<p class="muted">Denoising/results.json not found.</p>'; return; }
    const sig = [0, 15, 25, 50];
    const W = 560, H = 280, padL = 40, padR = 16, padT = 10, padB = 28;
    const x = v => padL + (v / 50) * (W - padL - padR);
    const y = v => padT + (1 - v) * (H - padT - padB);
    let s = `<svg viewBox="0 0 ${W} ${H}">`;
    for (let v = 0; v <= 1.0001; v += 0.2) s += `<line class="grid" x1="${padL}" x2="${W - padR}" y1="${y(v)}" y2="${y(v)}"/><text x="${padL - 6}" y="${y(v) + 4}" text-anchor="end">${Math.round(v * 100)}%</text>`;
    sig.forEach(v => (s += `<text x="${x(v)}" y="${H - 8}" text-anchor="middle">σ = ${v}</text>`));
    MODEL_ORDER.forEach(m => {
      const a = acc[MODEL_LABELS[m]];
      if (!a) return;
      const noisy = [a.clean, a.noisy_15, a.noisy_25, a.noisy_50];
      const den = [a.clean, a["fine-tuned_15"], a["fine-tuned_25"], a["fine-tuned_50"]];
      const col = MODEL_COLORS[m];
      const path = arr => arr.map((v, i) => `${i ? "L" : "M"}${x(sig[i])},${y(v)}`).join("");
      s += `<path d="${path(den)}" stroke="${col}" stroke-width="2.5"/>`;
      s += `<path d="${path(noisy)}" stroke="${col}" stroke-width="2" stroke-dasharray="5 4" opacity=".75"/>`;
      sig.forEach((sv, i) => {
        s += `<circle data-tip="<b>${MODEL_LABELS[m]}</b> · σ=${sv}<br>noisy ${pct(noisy[i])} · denoised ${pct(den[i])}" cx="${x(sv)}" cy="${y(den[i])}" r="4.5" fill="${col}" stroke="var(--panel)" stroke-width="2"/>`;
        if (i) s += `<circle data-tip="<b>${MODEL_LABELS[m]}</b> · σ=${sv}<br>noisy ${pct(noisy[i])}" cx="${x(sv)}" cy="${y(noisy[i])}" r="3.5" fill="var(--panel)" stroke="${col}" stroke-width="2"/>`;
      });
    });
    s += "</svg>";
    el.innerHTML = s + legendHTML([...MODEL_ORDER.map(m => [MODEL_LABELS[m], MODEL_COLORS[m]]), ["Noisy input", "#8a90a2", true]]);
    attachTips(el);
  },

  psnr() {
    const q = state.info?.results?.denoising_quality;
    if (!q) { $("#psnrTable").innerHTML = '<p class="muted">Denoising/results.json not found.</p>'; return; }
    const rows = Object.keys(q).map(sg => {
      const r = q[sg];
      const cell = (k, best) => `<td class="${best ? "best" : ""}">${r[k].psnr.toFixed(2)} / ${r[k].ssim.toFixed(3)}</td>`;
      return `<tr><td>σ = ${sg}</td>${cell("noisy")}${cell("pretrained")}${cell("fine-tuned", true)}</tr>`;
    }).join("");
    $("#psnrTable").innerHTML = `<table class="data"><thead><tr><th>Noise</th><th>Noisy input</th><th>Pretrained DnCNN</th><th>Fine-tuned DnCNN</th></tr></thead><tbody>${rows}</tbody></table>`;
  },

  balance() {
    const W = 560, rowH = 24, padL = 70, padR = 50;
    const H = EMOTIONS.length * rowH;
    const max = Math.max(...BENCH.trainCounts);
    const x = v => padL + (v / max) * (W - padL - padR);
    let s = `<svg viewBox="0 0 ${W} ${H}">`;
    EMOTIONS.forEach((e, i) => {
      const y = i * rowH;
      const v = BENCH.trainCounts[i];
      s += `<text class="lbl" x="0" y="${y + rowH / 2}" dominant-baseline="middle">${e}</text>`;
      s += `<rect class="hit" data-tip="${e}: ${v.toLocaleString()} train · ${BENCH.testCounts[i]} test" x="${padL}" y="${y + 4}" width="${x(v) - padL}" height="${rowH - 8}" rx="3" fill="${EMO_COLORS[i]}" stroke="none"/>`;
      s += `<text class="val" x="${x(v) + 6}" y="${y + rowH / 2}" dominant-baseline="middle">${v.toLocaleString()}</text>`;
    });
    s += "</svg>";
    $("#balanceChart").innerHTML = s + `<p class="hint" style="margin-top:8px">Happy has 17× more training images than Fear, the weakest class for every model.</p>`;
    attachTips($("#balanceChart"));
  },
};

/* ================= Boot ================= */
async function boot() {
  bindSettings();
  Gan.init();
  bindTabs();
  Live.init();
  Img.init();
  try {
    const info = await (await fetch("/api/info")).json();
    state.info = info;
    const chip = $("#deviceChip");
    chip.classList.add("ok");
    chip.lastElementChild.textContent = info.device.toUpperCase() + " · ready";

    $("#weightsSel").innerHTML = Object.entries(info.weight_sets)
      .filter(([k]) => info.available[k].length)
      .map(([k, v]) => `<option value="${k}">${esc(v)}</option>`).join("");
    state.weights = $("#weightsSel").value || "current";
    refreshModelAvailability();
    $$("#modelSeg button").forEach(b => {
      const acc = BENCH.models[b.dataset.v]?.acc;
      if (acc) b.innerHTML = `${MODEL_LABELS[b.dataset.v]}<span class="acc">${acc.toFixed(1)}%</span>`;
    });
    if (!info.denoiser) {
      $("#denoiseRow").classList.add("disabled");
      $("#denoiseRow small").textContent = "Weights not found";
    }
    if (!info.samples) {
      ["#sampleBtn", "#sampleBtn2"].forEach(s => ($(s).hidden = true));
      $("#sampleChips").hidden = true;
    }
    if ($("#view-bench").classList.contains("active")) { Bench.rendered = false; Bench.render(); }
    if (!info.gan) Gan.showMissing();
    if ($("#view-gan").classList.contains("active")) Gan.renderStats();
  } catch {
    const chip = $("#deviceChip");
    chip.classList.add("err");
    chip.lastElementChild.textContent = "server offline";
  }
}

boot();
