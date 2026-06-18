# orient_review/review_tool.py
"""文本行方向复核工具(OCR 识别数据集版)。

逐张复核方向模型挑出的"疑似上下颠倒"小图，人工确认后原地旋转 180° 覆盖,
覆盖前自动备份，可撤销。识别标签(文本)与方向无关，全程不动 train/val.txt。

用法:
  1. python predict_candidates.py   # 先生成 candidates.json
  2. python review_tool.py          # 启动，浏览器开 http://127.0.0.1:5001
"""
import os

from flask import Flask, request, jsonify, send_file, abort, render_template_string

import review_io as rio

app = Flask(__name__)

PAGE = """<!doctype html>
<html lang="zh"><head><meta charset="utf-8"><title>方向复核</title>
<style>
 body{font-family:sans-serif;margin:0;background:#f5f5f5}
 #bar{position:sticky;top:0;background:#fff;padding:10px 16px;border-bottom:1px solid #ddd;z-index:10}
 #bar button{margin-right:6px;padding:4px 10px}
 #grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:12px;padding:16px}
 .card{background:#fff;border:1px solid #ddd;border-radius:6px;padding:8px;text-align:center}
 .card.done{outline:3px solid #2e7d32}
 .card img{max-width:100%;max-height:130px;cursor:zoom-in;background:#eee}
 .meta{font-size:13px;margin:6px 0;color:#444}
 .apply{background:#1565c0;color:#fff;border:none;border-radius:4px;cursor:pointer}
 .undo{background:#eee;border:1px solid #bbb;border-radius:4px;cursor:pointer}
 .path{font-size:11px;color:#999;word-break:break-all}
 .hint{color:#888;font-size:12px}
</style></head><body>
<div id="bar">
  <b>候选 <span id="ncand">{{n}}</span></b> · 已翻转 <span id="done">{{ndone}}</span> · 剩余 <span id="left">{{nleft}}</span>
  &nbsp;<span id="scan" class="hint"></span>
  &nbsp;|&nbsp; 筛选:
  <button onclick="flt('all')">全部</button>
  <button onclick="flt('todo')">仅未处理</button>
  <button onclick="flt('done')">仅已翻转</button>
  &nbsp;|&nbsp;
  <button class="apply" onclick="applyVisible()">翻转当前可见的未处理项</button>
  &nbsp;<span class="hint">逐张点「确认翻转」，或先筛选/核查后用右侧按钮批量翻转可见项；均自动备份可撤销</span>
</div>
<div id="grid">
{% for e in cands %}
  <div class="card {{'done' if e.status=='rotated' else ''}}" data-path="{{e.path}}" data-status="{{e.status}}">
    <img src="/image?path={{e.path|urlencode}}&v={{e.v}}" onclick="window.open(this.src)">
    <div class="meta">疑似反向 · score {{e.score}}</div>
    <div class="act">
      <button class="apply" onclick="apply(this)" style="display:{{'none' if e.status=='rotated' else 'inline'}}">确认翻转 180°</button>
      <button class="undo" onclick="undo(this)" style="display:{{'inline' if e.status=='rotated' else 'none'}}">撤销</button>
    </div>
    <div class="path">{{e.fname}}</div>
  </div>
{% endfor %}
</div>
<script>
function counts(){
  let cs=[...document.querySelectorAll('.card')];
  let d=cs.filter(c=>c.dataset.status==='rotated').length;
  document.getElementById('ncand').textContent=cs.length;
  document.getElementById('done').textContent=d;
  document.getElementById('left').textContent=cs.length-d;
}
function esc(s){let d=document.createElement('div');d.textContent=s;return d.innerHTML;}
function buildCard(e){ // 与服务端模板一致，供轮询追加新候选
  let div=document.createElement('div');
  div.className='card'+(e.status==='rotated'?' done':'');
  div.dataset.path=e.path; div.dataset.status=e.status||'';
  let v=e.status==='rotated'?Date.now():0;
  div.innerHTML=
    '<img src="/image?path='+encodeURIComponent(e.path)+'&v='+v+'" onclick="window.open(this.src)">'+
    '<div class="meta">疑似反向 · score '+esc(''+e.score)+'</div>'+
    '<div class="act">'+
      '<button class="apply" onclick="apply(this)" style="display:'+(e.status==='rotated'?'none':'inline')+'">确认翻转 180°</button>'+
      '<button class="undo" onclick="undo(this)" style="display:'+(e.status==='rotated'?'inline':'none')+'">撤销</button>'+
    '</div><div class="path">'+esc(e.fname)+'</div>';
  return div;
}
let seen=new Set([...document.querySelectorAll('.card')].map(c=>c.dataset.path));
function poll(){
  fetch('/candidates').then(r=>r.json()).then(d=>{
    let grid=document.getElementById('grid');
    d.candidates.forEach(e=>{ if(!seen.has(e.path)){ seen.add(e.path); grid.appendChild(buildCard(e)); }});
    let s=d.scan||{};
    document.getElementById('scan').textContent = s.finished
      ? '✓ 扫描完成 ('+(s.done||0)+' 张已扫)'
      : '⏳ 扫描中 '+(s.done||0)+'/'+(s.total||'?')+' …页面会自动追加新候选';
    counts();
    if(!s.finished) setTimeout(poll,3000); // 扫完即停止轮询
  }).catch(()=>setTimeout(poll,5000));
}
poll();
function bust(card){ // 强制刷新该卡缩略图(图片已被改写)
  let img=card.querySelector('img'); let u=new URL(img.src,location.href);
  u.searchParams.set('v',Date.now()); img.src=u.pathname+u.search;
}
function apply(btn){
  let card=btn.closest('.card');
  fetch('/apply',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({path:card.dataset.path})}).then(r=>r.json()).then(d=>{
      if(!d.ok){alert(d.error);return;}
      card.dataset.status='rotated'; card.classList.add('done');
      btn.style.display='none'; card.querySelector('.undo').style.display='inline';
      bust(card); counts(); });
}
function undo(btn){
  let card=btn.closest('.card');
  fetch('/undo',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({path:card.dataset.path})}).then(r=>r.json()).then(d=>{
      if(!d.ok){alert(d.error);return;}
      card.dataset.status=''; card.classList.remove('done');
      btn.style.display='none'; card.querySelector('.apply').style.display='inline';
      bust(card); counts(); });
}
function flt(m){
  document.querySelectorAll('.card').forEach(c=>{
    let s=c.dataset.status==='rotated', show=true;
    if(m==='todo') show=!s; else if(m==='done') show=s;
    c.style.display=show?'':'none'; });
}
function applyVisible(){
  let cards=[...document.querySelectorAll('.card')].filter(
    c=>c.style.display!=='none' && c.dataset.status!=='rotated');
  if(!cards.length){alert('当前没有可翻转的未处理项');return;}
  if(!confirm('将批量翻转 '+cards.length+' 张可见的未处理图片，继续？'))return;
  let paths=cards.map(c=>c.dataset.path);
  fetch('/apply_batch',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({paths})}).then(r=>r.json()).then(d=>{
      if(!d.ok){alert(d.error||'批量失败');return;}
      let ok=new Set(d.rotated);
      cards.forEach(card=>{ if(ok.has(card.dataset.path)){
        card.dataset.status='rotated'; card.classList.add('done');
        card.querySelector('.apply').style.display='none';
        card.querySelector('.undo').style.display='inline'; bust(card); }});
      counts();
      if(d.failed && d.failed.length) alert('有 '+d.failed.length+' 张失败，详见控制台');
    });
}
</script>
</body></html>"""


def _candidates_with_state():
    try:
        cands = rio.load_candidates()
    except FileNotFoundError:
        cands = []  # 扫描还没开始/没产出候选文件
    state = rio.load_state()
    out = []
    for c in cands:
        path = c["path"]
        status = state.get(path, "")
        out.append({
            "path": path,
            "fname": os.path.basename(path),
            "score": c.get("score", ""),
            "status": status,
            # 已翻转的图加版本号，避免浏览器缓存显示翻转前的图
            "v": 1 if status == "rotated" else 0,
        })
    return out


@app.route("/")
def index():
    cands = _candidates_with_state()
    ndone = sum(1 for c in cands if c["status"] == "rotated")
    return render_template_string(PAGE, cands=cands, n=len(cands),
                                  ndone=ndone, nleft=len(cands) - ndone)


@app.route("/candidates")
def candidates():
    """供前端轮询：返回当前全部候选(含复核状态)与扫描进度，实现边扫边显示。"""
    return jsonify(candidates=_candidates_with_state(), scan=rio.load_scan())


@app.route("/image")
def image():
    # 放行依据 = 候选清单里实际存在的图，而非写死目录。
    # 这样换 --images-dir 重扫即自动生效，且只可能读到候选图本身(防穿越)。
    path = request.args.get("path", "")
    real = os.path.realpath(path)
    allowed = {os.path.realpath(c["path"]) for c in _load_candidates_safe()}
    if real not in allowed or not os.path.isfile(real):
        abort(404)
    resp = send_file(real)
    resp.headers["Cache-Control"] = "no-store"
    return resp


def _load_candidates_safe():
    try:
        return rio.load_candidates()
    except FileNotFoundError:
        return []


def _is_candidate(path):
    return any(c["path"] == path for c in _load_candidates_safe())


@app.route("/apply", methods=["POST"])
def apply():
    path = (request.get_json(silent=True) or {}).get("path")
    if not _is_candidate(path):
        return jsonify(ok=False, error="未找到候选"), 400
    try:
        rio.rotate_180(path)
    except Exception as ex:
        return jsonify(ok=False, error=str(ex)), 400
    state = rio.load_state()
    state[path] = "rotated"
    rio.save_state(state)
    return jsonify(ok=True)


@app.route("/apply_batch", methods=["POST"])
def apply_batch():
    paths = (request.get_json(silent=True) or {}).get("paths") or []
    cand_set = {c["path"] for c in _load_candidates_safe()}
    state = rio.load_state()
    rotated, failed = [], []
    for path in paths:
        if path not in cand_set or state.get(path) == "rotated":
            continue
        try:
            rio.rotate_180(path)
            state[path] = "rotated"
            rotated.append(path)
        except Exception as ex:
            failed.append({"path": path, "error": str(ex)})
    rio.save_state(state)
    return jsonify(ok=True, rotated=rotated, failed=failed)


@app.route("/undo", methods=["POST"])
def undo():
    path = (request.get_json(silent=True) or {}).get("path")
    if not _is_candidate(path):
        return jsonify(ok=False, error="未找到候选"), 400
    state = rio.load_state()
    if state.get(path) != "rotated":
        return jsonify(ok=False, error="该候选未翻转过，无需撤销"), 400
    try:
        rio.restore(path)
    except Exception as ex:
        return jsonify(ok=False, error=str(ex)), 400
    state.pop(path, None)
    rio.save_state(state)
    return jsonify(ok=True)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=False)
