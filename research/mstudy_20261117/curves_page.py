"""Training-curves page: inline SVG line charts with crosshair tooltip, direct labels, legend and table view."""
import json, os
S = os.path.dirname(os.path.abspath(__file__))

CHART_CSS = r"""
.viz{--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#eda100;--s5:#e87ba4;--grid:#e3e6eb;--axis:#8a93a0}
@media (prefers-color-scheme: dark){:root:where(:not([data-theme="light"])) .viz{--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--s5:#d55181;--grid:#2a3441;--axis:#6b7684}}
:root[data-theme="dark"] .viz{--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--s5:#d55181;--grid:#2a3441;--axis:#6b7684}
.viz{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:14px 16px 10px;margin:14px 0 22px}
.viz .hdr{display:flex;flex-wrap:wrap;gap:10px 18px;align-items:baseline;justify-content:space-between}
.viz .ttl{font:600 15px/1.3 var(--f-display)}
.viz .sub{color:var(--muted);font-size:12.5px;margin-top:2px;max-width:70ch}
.seg{display:inline-flex;border:1px solid var(--line);border-radius:7px;overflow:hidden}
.seg button{font:500 12px/1 var(--f-mono);padding:7px 10px;border:0;background:transparent;color:var(--muted);cursor:pointer}
.seg button[aria-pressed="true"]{background:var(--code);color:var(--ink)}
.seg button:focus-visible{outline:2px solid var(--accent);outline-offset:-2px}
.legend{display:flex;flex-wrap:wrap;gap:6px 16px;margin:10px 0 4px;font-size:12.5px;color:var(--muted)}
.legend span{display:inline-flex;align-items:center;gap:6px}
.legend i{display:inline-block;width:16px;height:0;border-top:2px solid}
.plot{position:relative;overflow-x:auto}
.plot svg{display:block;width:100%;height:auto;min-width:520px}
.plot text{fill:var(--muted);font:11px var(--f-mono)}
.plot .dl{font:600 11px var(--f-body);fill:var(--ink)}
.tip{position:absolute;pointer-events:none;background:var(--surface);border:1px solid var(--line);border-radius:8px;
  padding:8px 10px;font-size:12px;box-shadow:0 4px 14px rgba(0,0,0,.12);min-width:180px;display:none;z-index:2}
.tip .x{color:var(--muted);font:11px var(--f-mono);margin-bottom:4px}
.tip .r{display:flex;align-items:center;gap:8px;justify-content:space-between}
.tip .r b{font-variant-numeric:tabular-nums;color:var(--ink)}
.tip .r span{display:inline-flex;align-items:center;gap:6px;color:var(--muted)}
.tip .r i{display:inline-block;width:12px;border-top:2px solid}
details.tv{margin-top:6px} details.tv summary{cursor:pointer;color:var(--muted);font-size:12.5px}
"""

CHART_JS = r"""
(function(){
const DATA = JSON.parse(document.getElementById('curve-data').textContent);
const COLORS=['var(--s1)','var(--s2)','var(--s3)','var(--s4)','var(--s5)'];
const NS='http://www.w3.org/2000/svg';
function el(t,a,p){const e=document.createElementNS(NS,t);for(const k in a)e.setAttribute(k,a[k]);if(p)p.appendChild(e);return e;}
function nice(lo,hi,n){const span=hi-lo||1;const step=Math.pow(10,Math.floor(Math.log10(span/n)));const err=span/n/step;
  const m=err>=7.5?10:err>=3.5?5:err>=1.5?2:1;const s=m*step;const a=Math.floor(lo/s)*s,b=Math.ceil(hi/s)*s;const t=[];for(let v=a;v<=b+s/2;v+=s)t.push(+v.toFixed(10));return t;}
function draw(box){
  const spec=JSON.parse(box.dataset.spec);
  const plot=box.querySelector('.plot'); const tip=box.querySelector('.tip'); const tbody=box.querySelector('tbody'); const thead=box.querySelector('thead');
  let hz=0;
  function seriesNow(){return spec.series.map((s,i)=>({name:s.name,color:COLORS[i],pts:s.pts.map(p=>({x:p.x,y:Array.isArray(p.y)?p.y[hz]:p.y,resume:p.resume,secs:p.secs}))}));}
  function render(){
    plot.querySelectorAll('svg').forEach(n=>n.remove());
    const ser=seriesNow(); const W=760,H=300,L=56,R=150,T=12,B=40;
    const xs=ser.flatMap(s=>s.pts.map(p=>p.x)), ys=ser.flatMap(s=>s.pts.map(p=>p.y));
    const xt=nice(spec.x0??Math.min(...xs),Math.max(...xs),6); const pad=(Math.max(...ys)-Math.min(...ys))*.12||0.01;
    const yt=nice(Math.min(...ys)-pad,Math.max(...ys)+pad,5);
    const X=v=>L+(v-xt[0])/(xt[xt.length-1]-xt[0])*(W-L-R), Y=v=>T+(1-(v-yt[0])/(yt[yt.length-1]-yt[0]))*(H-T-B);
    const svg=el('svg',{viewBox:`0 0 ${W} ${H}`,role:'img','aria-label':spec.title});
    yt.forEach(v=>{el('line',{x1:L,x2:W-R,y1:Y(v),y2:Y(v),stroke:'var(--grid)','stroke-width':1},svg);const t=el('text',{x:L-8,y:Y(v)+4,'text-anchor':'end'},svg);t.textContent=spec.yfmt==='pct'?v.toFixed(2):(+v.toFixed(4)).toString();});
    xt.forEach(v=>{const t=el('text',{x:X(v),y:H-B+18,'text-anchor':'middle'},svg);t.textContent=(+v.toFixed(3)).toString();});
    el('line',{x1:L,x2:W-R,y1:H-B,y2:H-B,stroke:'var(--axis)','stroke-width':1},svg);
    const xl=el('text',{x:(L+W-R)/2,y:H-4,'text-anchor':'middle'},svg); xl.textContent=spec.xlabel;
    const yl=el('text',{x:12,y:(T+H-B)/2,'text-anchor':'middle',transform:`rotate(-90 12 ${(T+H-B)/2})`},svg); yl.textContent=spec.ylabel+(spec.hz?` (${DATA.horizons[hz]})`:'');
    const labels=[];
    ser.forEach(s=>{
      if(s.pts.length>1) el('path',{d:s.pts.map((p,i)=>(i?'L':'M')+X(p.x).toFixed(1)+' '+Y(p.y).toFixed(1)).join(' '),fill:'none',stroke:s.color,'stroke-width':2,'stroke-linejoin':'round','stroke-linecap':'round'},svg);
      s.pts.forEach(p=>{el('circle',{cx:X(p.x),cy:Y(p.y),r:p.resume?6:4,fill:p.resume?'var(--surface)':s.color,stroke:p.resume?s.color:'var(--surface)','stroke-width':p.resume?2.5:2},svg);});
      const last=s.pts[s.pts.length-1]; labels.push({y:Y(last.y),x:X(last.x),name:s.name,v:last.y});
    });
    labels.sort((a,b)=>a.y-b.y); for(let i=1;i<labels.length;i++) if(labels[i].y-labels[i-1].y<13) labels[i].y=labels[i-1].y+13;
    labels.forEach(l=>{const t=el('text',{x:W-R+10,y:l.y+4,class:'dl'},svg);t.textContent=l.name.length>24?l.name.slice(0,23)+'…':l.name;});
    const res=ser.flatMap(s=>s.pts.filter(p=>p.resume));
    res.forEach(p=>{const t=el('text',{x:X(p.x),y:Y(p.y)-12,'text-anchor':'middle'},svg);t.textContent='resumed';});
    const hair=el('line',{y1:T,y2:H-B,stroke:'var(--axis)','stroke-width':1,'stroke-dasharray':'3 3',visibility:'hidden'},svg);
    const hit=el('rect',{x:L,y:T,width:W-L-R,height:H-T-B,fill:'transparent',tabindex:0},svg);
    const allx=[...new Set(xs)].sort((a,b)=>a-b);
    function show(px){
      const vx=xt[0]+(px-L)/(W-L-R)*(xt[xt.length-1]-xt[0]);
      let nx=allx[0];allx.forEach(x=>{if(Math.abs(x-vx)<Math.abs(nx-vx))nx=x;});
      hair.setAttribute('x1',X(nx));hair.setAttribute('x2',X(nx));hair.setAttribute('visibility','visible');
      tip.replaceChildren();const hx=document.createElement('div');hx.className='x';hx.textContent=`${spec.xlabel}: ${nx}`;tip.appendChild(hx);
      const tol=(xt[xt.length-1]-xt[0])*0.06;
      ser.forEach(s=>{let best=null;s.pts.forEach(p=>{if(Math.abs(p.x-nx)<=tol&&(!best||Math.abs(p.x-nx)<Math.abs(best.x-nx)))best=p;});
        const r=document.createElement('div');r.className='r';const sp=document.createElement('span');const i=document.createElement('i');i.style.borderColor=s.color;sp.appendChild(i);sp.appendChild(document.createTextNode(s.name));
        const b=document.createElement('b');b.textContent=best?best.y.toFixed(4):'–';r.appendChild(sp);r.appendChild(b);tip.appendChild(r);});
      const bb=plot.getBoundingClientRect(), sx=bb.width/W; tip.style.display='block';
      let left=X(nx)*sx+12; if(left+tip.offsetWidth>bb.width) left=X(nx)*sx-tip.offsetWidth-12; tip.style.left=left+'px'; tip.style.top='8px';}
    hit.addEventListener('pointermove',e=>{const bb=svg.getBoundingClientRect();show((e.clientX-bb.left)/bb.width*W);});
    hit.addEventListener('pointerleave',()=>{tip.style.display='none';hair.setAttribute('visibility','hidden');});
    hit.addEventListener('focus',()=>show(X(allx[allx.length-1])));
    hit.addEventListener('blur',()=>{tip.style.display='none';hair.setAttribute('visibility','hidden');});
    plot.insertBefore(svg,tip);
    // table
    thead.replaceChildren(); tbody.replaceChildren();
    const hr=document.createElement('tr');['series',spec.xlabel,spec.ylabel+(spec.hz?' ('+DATA.horizons[hz]+')':'')].forEach(h=>{const th=document.createElement('th');th.textContent=h;hr.appendChild(th);});thead.appendChild(hr);
    ser.forEach(s=>s.pts.forEach(p=>{const tr=document.createElement('tr');[s.name+(p.resume?' (resume point)':''),p.x,p.y.toFixed(4)].forEach(v=>{const td=document.createElement('td');td.textContent=v;tr.appendChild(td);});tbody.appendChild(tr);}));
  }
  const lg=box.querySelector('.legend'); spec.series.forEach((s,i)=>{const sp=document.createElement('span');const ii=document.createElement('i');ii.style.borderColor=COLORS[i];sp.appendChild(ii);sp.appendChild(document.createTextNode(s.name));lg.appendChild(sp);});
  box.querySelectorAll('.seg button').forEach(b=>b.addEventListener('click',()=>{hz=+b.dataset.h;box.querySelectorAll('.seg button').forEach(x=>x.setAttribute('aria-pressed',x===b?'true':'false'));render();}));
  render(); window.addEventListener('resize',()=>{tip.style.display='none';});
}
document.querySelectorAll('.viz[data-spec]').forEach(draw);
})();
"""


def chart(cid, title, sub, spec):
    seg = ''
    if spec.get('hz'):
        seg = '<div class="seg" role="group" aria-label="Horizon">' + ''.join(
            f'<button type="button" id="{cid}-h{i}" data-h="{i}" aria-pressed="{"true" if i == 0 else "false"}">{h}</button>'
            for i, h in enumerate(['0.1s', '0.5s', '1s', '10s'])) + '</div>'
    js = json.dumps(spec).replace("'", '&#39;')
    return (f'<div class="viz" id="{cid}" data-spec=\'{js}\'><div class="hdr"><div><div class="ttl">{title}</div>'
            f'<div class="sub">{sub}</div></div>{seg}</div><div class="legend"></div>'
            f'<div class="plot"><div class="tip" role="status"></div></div>'
            f'<details class="tv"><summary>Show data table</summary><div class="tbl"><table><thead></thead><tbody></tbody></table></div></details></div>')


def body():
    D = json.load(open(f'{S}/curves_data.json'))
    f = D['fair']
    order15 = ['Transformer · 40 min', 'GRU · 30 min', 'Transformer · 75 min + continued', 'GRU · 60 min']
    sp15 = dict(title='BTC 15m validation R²', xlabel='training samples seen (millions)', ylabel='validation R²', hz=True, x0=0,
                series=[dict(name=n, pts=[dict(x=p['x'], y=p['r2'], resume=p.get('resume', False), secs=p['secs']) for p in f['BTC15M'][n]])
                        for n in order15 if n in f['BTC15M']])
    spp = dict(title='BTC perp validation R²', xlabel='training samples seen (millions)', ylabel='validation R²', hz=True, x0=0,
               series=[dict(name=n, pts=[dict(x=p['x'], y=p['r2'], secs=p['secs']) for p in f['BTCPERP'][n]]) for n in f['BTCPERP']])
    o = D['other']
    spae = dict(title='Autoencoder', xlabel='epoch', ylabel='validation MSE', series=[dict(name=k, pts=[dict(x=p['x'], y=p['y']) for p in v]) for k, v in o['ae'].items()])
    spcl = dict(title='Market-type classifiers', xlabel='epoch', ylabel='validation balanced accuracy', yfmt='pct',
                series=[dict(name=k, pts=[dict(x=p['x'], y=p['y']) for p in v]) for k, v in o['clf'].items()])
    sptr = dict(title='Trade model', xlabel='epoch', ylabel='val toxic AUC (mean of 4 horizons)', series=[dict(name=k, pts=[dict(x=p['x'], y=p['y']) for p in v]) for k, v in o['trade'].items()])
    t = D['tests']
    def tr(m, k):
        v = t.get(f'{m}|{k}')
        return '' if not v else '<td>' + '</td><td>'.join(f'{x:.4f}' for x in v) + '</td>'
    rows = [('BTC 15m', 'LightGBM (handcrafted)', tr('BTC15M', 'LightGBM (handcrafted)')),
            ('BTC 15m', 'Transformer · 40 min', tr('BTC15M', 'transformer (raw events)')),
            ('BTC 15m', 'Transformer · 75 min', tr('BTC15M', 'transformer_long (raw events)')),
            ('BTC 15m', 'Transformer · 75 min + continued', tr('BTC15M', 'transformer_cont (raw events)')),
            ('BTC 15m', 'GRU · 30 min', tr('BTC15M', 'gru (raw events)')),
            ('BTC 15m', 'GRU · 60 min', tr('BTC15M', 'gru_long (raw events)')),
            ('BTC perp', 'LightGBM (handcrafted)', tr('BTCPERP', 'LightGBM (handcrafted)')),
            ('BTC perp', 'Transformer · 25 min', tr('BTCPERP', 'transformer (raw events)')),
            ('BTC perp', 'GRU · 20 min', tr('BTCPERP', 'gru (raw events)'))]
    ttab = ('<div class="tbl"><table><thead><tr><th>market</th><th>model</th><th>0.1s</th><th>0.5s</th><th>1s</th><th>10s</th></tr></thead><tbody>'
            + ''.join(f'<tr><td>{a}</td><td>{b}</td>{c}</tr>' if c else f'<tr><td>{a}</td><td>{b}</td><td colspan="4" style="text-align:left">still training</td></tr>' for a, b, c in rows)
            + '</tbody></table></div>')
    h = ('<h2>Raw-event fair models</h2>'
         + chart('c15', 'BTC 15m: validation R² while training',
                 'Each point is one validation pass on Sep 22 (unseen markets). The 75-minute Transformer was resumed from its saved weights at the hollow point and kept training (batch 512, 30 threads). Pick a horizon above the chart.', sp15)
         + chart('cperp', 'BTC perp: validation R² while training', 'The perp Transformer got only one validation pass in its 25-minute budget.', spp)
         + '<h3>Held-out test R² (Sep 23) at the end of each run</h3>' + ttab
         + '<h2>Other training curves</h2>'
         + chart('cae', 'Autoencoder reconstruction', 'Validation MSE per epoch; the mean-prediction baseline is 1.29 (main) on the same standardised tokens. Lower is better.', spae)
         + chart('ccl', 'Market-type classifiers', 'Validation balanced accuracy per epoch for the 13-class task (chance 0.08).', spcl)
         + chart('ctr', 'Trade toxicity model', 'Validation toxic-flow AUC per epoch for each input set (early stopping on this curve).', sptr)
         + '<p class="note">Fair-model points come from the training log on the VM, which records every validation pass. Several of those runs were first synced to wandb while still training, and wandb kept only the part synced at that time, so their wandb charts are incomplete. The autoencoder, classifier and trade curves come straight from wandb.</p>')
    return h, CHART_CSS, CHART_JS, json.dumps({'horizons': D['horizons']})
