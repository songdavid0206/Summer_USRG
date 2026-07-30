"""Build a self-contained run-by-run Laguna Seca distribution viewer."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "analysis" / "lagunaseca_distribution"
OUTPUT = ANALYSIS / "run_viewer.html"
REFERENCE = (
    ROOT
    / "dataset_v2"
    / "solo"
    / "20260724_154559_lagunaseca_ferrarisf15t_clear_solo_40hz"
    / "TRACK_INFO"
    / "AI_RACING_LINE.parquet"
)


def short_label(name: str) -> str:
    parts = name.split("_")
    stamp = f"{parts[0][4:]}-{parts[1]}"
    weather = next(
        (x for x in ("clear", "scatteredclouds", "mist") if x in parts),
        "clear",
    )
    mode = "MULTI" if "multi" in parts else "SOLO"
    aggression = parts[-1] if mode == "MULTI" and parts[-1].isdigit() else ""
    return f"{stamp} · {mode} · {weather}{(' · A' + aggression) if aggression else ''}"


def finite(value: object, default: float = 0.0) -> float:
    try:
        result = float(value)
        return result if result == result else default
    except Exception:
        return default


def build_payload() -> dict[str, object]:
    profiles = pd.read_csv(ANALYSIS / "track_bin_profiles.csv")
    sessions = pd.read_csv(ANALYSIS / "session_summary.csv").set_index("session")
    offsets = pd.read_csv(ANALYSIS / "run_lateral_offset_summary.csv").set_index("session")
    lap_summaries = pd.read_csv(ANALYSIS / "lap_summary.csv")
    lap_profiles = pd.read_csv(ANALYSIS / "lap_frame_profiles.csv")
    reference = pd.read_parquet(REFERENCE, columns=["x", "y"]).iloc[::4]

    runs: dict[str, object] = {}
    order = sessions.index.tolist()
    for name in order:
        summary = sessions.loc[name]
        offset = offsets.loc[name]
        points = profiles[profiles["session"] == name].sort_values("track_bin")
        session_lap_summaries = lap_summaries[lap_summaries["session"] == name]
        session_lap_profiles = lap_profiles[lap_profiles["session"] == name]
        lap_payload = {}
        for lap_row in session_lap_summaries.itertuples(index=False):
            lap_number = int(lap_row.lap)
            lap_points = session_lap_profiles[
                session_lap_profiles["lap"] == lap_number
            ].sort_values("track_bin")
            lap_payload[str(lap_number)] = {
                "summary": {
                    "frames": int(lap_row.frames),
                    "duration": finite(lap_row.duration_s),
                    "meanSpeed": finite(lap_row.mean_speed_kmh),
                    "meanError": finite(lap_row.mean_abs_lateral_offset_m),
                    "p95Error": finite(lap_row.p95_abs_lateral_offset_m),
                    "p99Error": finite(lap_row.p99_abs_lateral_offset_m),
                },
                "points": [
                    {
                        "p": finite(row.spline),
                        "x": finite(row.x),
                        "y": finite(row.y),
                        "speed": finite(row.speed_kmh),
                        "error": finite(row.abs_lateral_offset_m),
                    }
                    for row in lap_points.itertuples(index=False)
                ],
            }
        runs[name] = {
            "label": short_label(name),
            "mode": str(summary["mode"]),
            "weather": str(summary["weather"]),
            "summary": {
                "frames": int(summary["frames"]),
                "minutes": finite(summary["duration_s"]) / 60.0,
                "laps": int(summary["laps"]),
                "meanSpeed": finite(summary["mean_speed_kmh"]),
                "p95Speed": finite(summary["p95_speed_kmh"]),
                "meanError": finite(offset["mean"]),
                "p95Error": finite(offset["p95"]),
                "p99Error": finite(offset["p99"]),
                "gridGaps": int(summary["grid_gaps"]),
                "interactionFraction": finite(summary["interaction_fraction"]),
            },
            "points": [
                {
                    "p": (int(row.track_bin) + 0.5) / 100.0,
                    "x": finite(row.x),
                    "y": finite(row.y),
                    "speed": finite(row.speed_kmh),
                    "errorMean": finite(row.abs_lateral_offset_m),
                    "errorP95": finite(row.p95_abs_lateral_offset_m),
                }
                for row in points.itertuples(index=False)
            ],
            "laps": lap_payload,
        }
    return {
        "order": order,
        "runs": runs,
        "reference": [
            {"x": finite(row.x), "y": finite(row.y)}
            for row in reference.itertuples(index=False)
        ],
    }


HTML = r"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Laguna Seca Run Distribution Viewer</title>
<style>
:root {
  color-scheme: dark;
  --bg: #0e1117;
  --panel: #171c24;
  --panel2: #1d2430;
  --line: #303948;
  --text: #e9eef7;
  --muted: #9daabd;
  --accent: #65b8ff;
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--bg); color: var(--text);
  font-family: Inter, "Noto Sans KR", system-ui, sans-serif;
}
.wrap { max-width: 1540px; margin: 0 auto; padding: 22px; }
h1 { font-size: 25px; margin: 0 0 5px; }
.subtitle { color: var(--muted); margin-bottom: 18px; }
.toolbar {
  display: grid; grid-template-columns: auto minmax(340px,1fr) auto auto auto auto;
  gap: 9px; align-items: center; padding: 12px;
  background: var(--panel); border: 1px solid var(--line); border-radius: 12px;
}
button, select {
  color: var(--text); background: var(--panel2); border: 1px solid #3a4658;
  border-radius: 8px; padding: 9px 12px; font-size: 14px;
}
button { cursor: pointer; }
button:hover { border-color: var(--accent); }
.counter { color: var(--muted); min-width: 62px; text-align: center; }
.cards {
  display: grid; grid-template-columns: repeat(8, minmax(110px,1fr));
  gap: 9px; margin: 12px 0;
}
.card {
  background: var(--panel); border: 1px solid var(--line);
  border-radius: 10px; padding: 10px 12px;
}
.card .k { color: var(--muted); font-size: 11px; margin-bottom: 4px; }
.card .v { font-size: 18px; font-weight: 700; }
.grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
.panel {
  background: var(--panel); border: 1px solid var(--line);
  border-radius: 12px; padding: 12px;
}
.panelhead {
  display: flex; align-items: center; justify-content: space-between;
  gap: 10px; margin-bottom: 8px;
}
.panel h2 { font-size: 16px; margin: 0; }
.hint { color: var(--muted); font-size: 12px; }
canvas {
  width: 100%; height: auto; display: block; background: #f7f8fa;
  border-radius: 8px;
}
.charts { margin-top: 12px; }
.foot {
  color: var(--muted); line-height: 1.55; font-size: 12px; margin-top: 13px;
}
.solo { color: #6dbdff; } .multi { color: #ff9d76; }
@media (max-width: 1050px) {
  .grid { grid-template-columns: 1fr; }
  .cards { grid-template-columns: repeat(4,1fr); }
  .toolbar { grid-template-columns: auto 1fr auto; }
  .toolbar .secondary { display: none; }
}
</style>
</head>
<body>
<div class="wrap">
  <h1>Laguna Seca · Run Distribution Viewer</h1>
  <div class="subtitle">이미지 프레임과 정합된 주행 라벨 기반 · ←/→ 키로 run 이동</div>

  <div class="toolbar">
    <button id="prev">← 이전</button>
    <select id="runSelect"></select>
    <button id="next">다음 →</button>
    <select id="scopeSelect">
      <option value="all">전체 run 요약</option>
    </select>
    <select id="modeFilter" class="secondary">
      <option value="all">SOLO + MULTI</option>
      <option value="solo">SOLO만</option>
      <option value="multi">MULTI만</option>
    </select>
    <span class="counter" id="counter"></span>
  </div>

  <div class="cards" id="cards"></div>

  <div class="grid">
    <section class="panel">
      <div class="panelhead">
        <h2>실제 궤적 · 속도</h2>
        <span class="hint">파랑=저속 · 빨강=고속</span>
      </div>
      <canvas id="speedMap" width="900" height="560"></canvas>
    </section>
    <section class="panel">
      <div class="panelhead">
        <h2>실제 궤적 · AI 레이싱라인 절대 오차</h2>
        <div>
          <span id="actualErrorLabel" class="hint" style="display:none">프레임 실제 오차</span>
          <select id="errorStat">
            <option value="errorP95">구간별 p95 오차</option>
            <option value="errorMean">구간별 평균 오차</option>
          </select>
        </div>
      </div>
      <canvas id="errorMap" width="900" height="560"></canvas>
    </section>
  </div>

  <section class="panel charts">
    <div class="panelhead">
      <h2>트랙 진행도별 속도와 절대 오차</h2>
      <span class="hint">세로 점선은 가장 큰 오차 구간</span>
    </div>
    <canvas id="profileChart" width="1800" height="470"></canvas>
  </section>

  <div class="foot">
    검은 점선은 AI 레이싱라인, 색 실선은 해당 run의 실제 ego 궤적이다.
    절대 오차는 spline progress에 대응하는 AI 라인 점의 로컬 법선 방향 거리로 계산했다.
    멀티의 큰 오차는 트랙 이탈뿐 아니라 추월·회피를 위한 의도적 라인 변경도 포함한다.
  </div>
</div>

<script>
const DATA = __DATA__;
const select = document.getElementById("runSelect");
const scopeSelect = document.getElementById("scopeSelect");
const filter = document.getElementById("modeFilter");
const counter = document.getElementById("counter");
const errorStat = document.getElementById("errorStat");
const actualErrorLabel = document.getElementById("actualErrorLabel");
let visible = [...DATA.order];
let index = 0;

function filteredOrder() {
  const f = filter.value;
  return DATA.order.filter(n => f === "all" || DATA.runs[n].mode === f);
}
function populate(keepName) {
  visible = filteredOrder();
  select.innerHTML = "";
  visible.forEach((name, i) => {
    const option = document.createElement("option");
    option.value = name;
    option.textContent = DATA.runs[name].label;
    select.appendChild(option);
  });
  index = Math.max(0, keepName ? visible.indexOf(keepName) : 0);
  if (index < 0) index = 0;
  select.selectedIndex = index;
  populateScopes();
  render();
}
function populateScopes() {
  const run = DATA.runs[visible[index]];
  scopeSelect.innerHTML = '<option value="all">전체 run 요약</option>';
  Object.keys(run.laps).sort((a,b)=>Number(a)-Number(b)).forEach(lap => {
    const option = document.createElement("option");
    option.value = lap;
    option.textContent = `Lap ${lap} · 실제 프레임`;
    scopeSelect.appendChild(option);
  });
  scopeSelect.value = "all";
}
function fmt(v, digits=1) { return Number(v).toFixed(digits); }
function cards(run, lap) {
  const s = lap ? lap.summary : run.summary;
  const items = lap ? [
    ["범위", `Lap ${scopeSelect.value}`],
    ["날씨", run.weather],
    ["원본 프레임", s.frames.toLocaleString()],
    ["랩타임", `${fmt(s.duration,2)} s`],
    ["평균속도", `${fmt(s.meanSpeed)} km/h`],
    ["평균 |오차|", `${fmt(s.meanError,2)} m`],
    ["p95 |오차|", `${fmt(s.p95Error,2)} m`],
    ["p99 |오차|", `${fmt(s.p99Error,2)} m`],
  ] : [
    ["모드", run.mode.toUpperCase()],
    ["날씨", run.weather],
    ["프레임", s.frames.toLocaleString()],
    ["완전 랩", s.laps],
    ["평균속도", `${fmt(s.meanSpeed)} km/h`],
    ["평균 |오차|", `${fmt(s.meanError,2)} m`],
    ["p95 |오차|", `${fmt(s.p95Error,2)} m`],
    ["p99 |오차|", `${fmt(s.p99Error,2)} m`],
  ];
  document.getElementById("cards").innerHTML = items.map(([k,v]) =>
    `<div class="card"><div class="k">${k}</div><div class="v">${v}</div></div>`
  ).join("");
}
function bounds() {
  const xs = DATA.reference.map(p=>p.x), ys = DATA.reference.map(p=>p.y);
  return {minX:Math.min(...xs), maxX:Math.max(...xs),
          minY:Math.min(...ys), maxY:Math.max(...ys)};
}
const B = bounds();
function transform(canvas) {
  const pad=42, w=canvas.width, h=canvas.height;
  const sx=(w-2*pad)/(B.maxX-B.minX), sy=(h-2*pad)/(B.maxY-B.minY);
  const scale=Math.min(sx,sy);
  const usedW=(B.maxX-B.minX)*scale, usedH=(B.maxY-B.minY)*scale;
  const ox=(w-usedW)/2, oy=(h-usedH)/2;
  return p => [ox+(p.x-B.minX)*scale, h-(oy+(p.y-B.minY)*scale)];
}
function speedColor(v) {
  const t=Math.max(0,Math.min(1,(v-60)/(290-60)));
  return `hsl(${235*(1-t)}, 88%, ${48+7*(1-Math.abs(t-.5)*2)}%)`;
}
function errorColor(v) {
  const t=Math.max(0,Math.min(1,v/4));
  return `hsl(${53*(1-t)}, ${82+12*t}%, ${77-36*t}%)`;
}
function baseMap(canvas) {
  const ctx=canvas.getContext("2d");
  ctx.clearRect(0,0,canvas.width,canvas.height);
  ctx.fillStyle="#f7f8fa"; ctx.fillRect(0,0,canvas.width,canvas.height);
  return [ctx, transform(canvas)];
}
function drawReference(ctx, tf) {
  ctx.save(); ctx.strokeStyle="#111827"; ctx.lineWidth=2.0;
  ctx.setLineDash([6,5]); ctx.beginPath();
  DATA.reference.forEach((p,i) => {
    const [x,y]=tf(p); if(i===0)ctx.moveTo(x,y); else ctx.lineTo(x,y);
  });
  const [x0,y0]=tf(DATA.reference[0]); ctx.lineTo(x0,y0); ctx.stroke(); ctx.restore();
}
function legend(ctx, colorFn, min, max, unit) {
  const x=canvasSafeWidth(ctx)-220, y=28, w=170, h=13;
  const grad=ctx.createLinearGradient(x,0,x+w,0);
  for(let i=0;i<=10;i++) grad.addColorStop(i/10,colorFn(min+(max-min)*i/10));
  ctx.fillStyle=grad; ctx.fillRect(x,y,w,h);
  ctx.strokeStyle="#374151"; ctx.strokeRect(x,y,w,h);
  ctx.fillStyle="#111827"; ctx.font="13px system-ui";
  ctx.fillText(`${min}`,x,y+31); ctx.fillText(`${max} ${unit}`,x+w-46,y+31);
}
function canvasSafeWidth(ctx) { return ctx.canvas.width; }
function drawTrack(canvas, points, valueKey, colorFn, min, max, unit) {
  const [ctx,tf]=baseMap(canvas);
  ctx.lineCap="round"; ctx.lineJoin="round"; ctx.lineWidth=7;
  for(let i=1;i<points.length;i++) {
    const a=tf(points[i-1]), b=tf(points[i]);
    ctx.strokeStyle=colorFn(points[i][valueKey]);
    ctx.beginPath(); ctx.moveTo(a[0],a[1]); ctx.lineTo(b[0],b[1]); ctx.stroke();
  }
  if(points.length>1) {
    const a=tf(points[points.length-1]), b=tf(points[0]);
    ctx.strokeStyle=colorFn(points[0][valueKey]);
    ctx.beginPath(); ctx.moveTo(a[0],a[1]); ctx.lineTo(b[0],b[1]); ctx.stroke();
  }
  drawReference(ctx,tf);
  const peak=points.reduce((a,b)=>b[valueKey]>a[valueKey]?b:a,points[0]);
  const [px,py]=tf(peak);
  ctx.fillStyle="#fff"; ctx.strokeStyle="#111"; ctx.lineWidth=2;
  ctx.beginPath(); ctx.arc(px,py,6,0,Math.PI*2); ctx.fill(); ctx.stroke();
  legend(ctx,colorFn,min,max,unit);
}
function axes(ctx, box, yMax, label) {
  ctx.strokeStyle="#596579"; ctx.lineWidth=1;
  ctx.beginPath(); ctx.moveTo(box.x,box.y); ctx.lineTo(box.x,box.y+box.h);
  ctx.lineTo(box.x+box.w,box.y+box.h); ctx.stroke();
  ctx.fillStyle="#d8e0eb"; ctx.font="15px system-ui";
  ctx.fillText(label,box.x+8,box.y+20);
  ctx.fillStyle="#9daabd"; ctx.font="12px system-ui";
  [0,.25,.5,.75,1].forEach(t=>{
    const x=box.x+t*box.w;
    ctx.fillText(`${Math.round(t*100)}%`,x-10,box.y+box.h+20);
  });
  [0,.5,1].forEach(t=>{
    const y=box.y+box.h-t*box.h;
    ctx.fillText(fmt(t*yMax,0),box.x-38,y+4);
  });
}
function drawLine(ctx, points, key, box, yMax, color) {
  ctx.strokeStyle=color; ctx.lineWidth=3; ctx.beginPath();
  points.forEach((p,i)=>{
    const x=box.x+p.p*box.w, y=box.y+box.h-Math.min(1,p[key]/yMax)*box.h;
    if(i===0)ctx.moveTo(x,y); else ctx.lineTo(x,y);
  }); ctx.stroke();
}
function drawProfiles(canvas, points, errKey) {
  const ctx=canvas.getContext("2d");
  ctx.clearRect(0,0,canvas.width,canvas.height);
  ctx.fillStyle="#171c24"; ctx.fillRect(0,0,canvas.width,canvas.height);
  const left={x:75,y:34,w:canvas.width-115,h:150};
  const right={x:75,y:250,w:canvas.width-115,h:150};
  const errMax=Math.max(4,Math.ceil(Math.max(...points.map(p=>p[errKey]))));
  const errLabel=errKey==="errorP95" ? "p95 절대 오차 (m)" :
                 errKey==="errorMean" ? "평균 절대 오차 (m)" : "프레임 실제 절대 오차 (m)";
  axes(ctx,left,300,"속도 (km/h)");
  axes(ctx,right,errMax,errLabel);
  drawLine(ctx,points,"speed",left,300,"#65b8ff");
  drawLine(ctx,points,errKey,right,errMax,"#ff845e");
  const peak=points.reduce((a,b)=>b[errKey]>a[errKey]?b:a,points[0]);
  ctx.strokeStyle="#d8e0eb"; ctx.setLineDash([5,5]); ctx.lineWidth=1;
  [left,right].forEach(box=>{
    const x=box.x+peak.p*box.w;
    ctx.beginPath(); ctx.moveTo(x,box.y); ctx.lineTo(x,box.y+box.h); ctx.stroke();
  });
  ctx.setLineDash([]);
  ctx.fillStyle="#e9eef7"; ctx.font="13px system-ui";
  ctx.fillText(`최대 오차 구간: ${(peak.p*100).toFixed(1)}%`,left.x+left.w-170,20);
}
function render() {
  if(!visible.length) return;
  index=(index+visible.length)%visible.length;
  const name=visible[index], run=DATA.runs[name];
  const lap=scopeSelect.value==="all" ? null : run.laps[scopeSelect.value];
  const points=lap ? lap.points : run.points;
  const errKey=lap ? "error" : errorStat.value;
  select.selectedIndex=index;
  counter.textContent=`${index+1} / ${visible.length}`;
  errorStat.style.display=lap ? "none" : "inline-block";
  actualErrorLabel.style.display=lap ? "inline" : "none";
  cards(run,lap);
  drawTrack(document.getElementById("speedMap"),points,"speed",speedColor,60,290,"km/h");
  drawTrack(document.getElementById("errorMap"),points,errKey,errorColor,0,4,"m");
  drawProfiles(document.getElementById("profileChart"),points,errKey);
}
function changeRun(delta) {
  index=(index+delta+visible.length)%visible.length;
  populateScopes();
  render();
}
document.getElementById("prev").onclick=()=>changeRun(-1);
document.getElementById("next").onclick=()=>changeRun(1);
select.onchange=()=>{index=select.selectedIndex;populateScopes();render();};
scopeSelect.onchange=render;
filter.onchange=()=>populate();
errorStat.onchange=render;
document.addEventListener("keydown",e=>{
  if(e.key==="ArrowLeft") changeRun(-1);
  if(e.key==="ArrowRight") changeRun(1);
});
populate();
</script>
</body>
</html>
"""


def main() -> None:
    required = [
        ANALYSIS / "track_bin_profiles.csv",
        ANALYSIS / "session_summary.csv",
        ANALYSIS / "run_lateral_offset_summary.csv",
        ANALYSIS / "lap_summary.csv",
        ANALYSIS / "lap_frame_profiles.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Run tools/analyze_lagunaseca_distribution.py first. Missing: "
            + ", ".join(missing)
        )
    payload = build_payload()
    OUTPUT.write_text(
        HTML.replace(
            "__DATA__",
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        ),
        encoding="utf-8",
    )
    print(f"[viewer] {OUTPUT}")
    print("[open]   start analysis\\lagunaseca_distribution\\run_viewer.html")


if __name__ == "__main__":
    main()
