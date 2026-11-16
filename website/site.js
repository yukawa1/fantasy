'use strict';
const data = window.spectrum;
const byId = id => document.getElementById(id);
const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)');
const stages = [
  ['Broad Hβ', '#ff827c', 'The isolated broad Hβ fit gives an intrinsic FWHM of 6,625 km/s and an integrated flux of 3.91 × 10⁻¹⁴ erg s⁻¹ cm⁻².'],
  ['Narrow lines', '#80d9a5', 'Narrow Hβ sharpens the line core, while the [O III] doublet appears at longer wavelengths.'],
  ['Fe II', '#bb95eb', 'Many overlapping Fe II transitions add low, broad blends on both sides of Hβ.'],
  ['Host starlight', '#f4c171', 'The host galaxy lifts the continuum and introduces stellar absorption, including a trough beneath Hβ.'],
  ['Full fit', '#67d6de', 'The accretion disk, Balmer continuum and remaining broad lines complete the fitted spectrum in this window.']
];
const field = byId('layers');
stages.forEach(([name,color], index) => {
  const label = document.createElement('label');
  const radio = document.createElement('input');
  radio.type='radio'; radio.name='stage'; radio.value=index; radio.checked=index===0;
  const caption=document.createElement('span'), dot=document.createElement('i');
  dot.style.setProperty('--swatch',color);
  caption.append(dot, document.createTextNode(name)); label.append(radio, caption); field.append(label);
  radio.addEventListener('change', () => setStage(index));
});

function surface(canvas) {
  const {width,height}=canvas.getBoundingClientRect(), ratio=Math.min(devicePixelRatio||1,2);
  const px=Math.round(width*ratio), py=Math.round(height*ratio);
  if(canvas.width!==px || canvas.height!==py){canvas.width=px;canvas.height=py}
  const ctx=canvas.getContext('2d');ctx.setTransform(ratio,0,0,ratio,0,0);ctx.clearRect(0,0,width,height);
  return {ctx,width,height};
}
function trace(ctx, points, color, lineWidth, baseline, fill) {
  ctx.beginPath();points.forEach(([x,y],i)=>i?ctx.lineTo(x,y):ctx.moveTo(x,y));
  ctx.lineWidth=lineWidth;ctx.lineJoin='round';ctx.strokeStyle=color;ctx.stroke();
  if(fill){ctx.lineTo(points.at(-1)[0],baseline);ctx.lineTo(points[0][0],baseline);ctx.closePath();ctx.fillStyle=fill;ctx.fill()}
}
function labelAt(ctx, text, x, y, color, align='center'){
  ctx.font='10px ui-monospace, monospace';ctx.textAlign=align;ctx.fillStyle=color;ctx.fillText(text,x,y);
}
const crest=[0.806,0.798,0.798,0.79,0.79,0.782,0.782,0.774,0.774,0.766,0.766,0.758,0.758,0.75,0.75,0.734,0.718,0.702,0.694,0.677,0.661,0.653,0.637,0.621,0.605,0.597,0.581,0.565,0.556,0.532,0.5,0.5,0.524,0.573,0.613,0.645,0.661,0.677,0.677,0.685,0.694,0.702,0.702,0.71,0.718,0.726,0.726,0.734,0.742,0.75,0.75,0.758,0.766,0.766,0.774,0.782,0.79,0.79,0.798,0.806,0.806,0.806,0.798,0.798,0.774,0.726,0.677,0.621,0.621,0.645,0.758,0.823,0.823,0.831,0.831,0.831,0.831,0.831,0.831,0.815,0.774,0.734,0.677,0.605,0.516,0.0,0.145,0.427,0.613,0.79,0.806,0.823,0.823,0.831,0.855,0.863,0.879,0.895,0.911,0.927,0.944,0.96,0.976,0.992];
const returnColors=['#ff827c','#80d9a5','#bb95eb','#f4c171','#67d6de'];
const ribbonStatus=byId('ribbon-status'), toggle=byId('motion');
const mark=new Image();mark.src='assets/mark.png';
let seconds=reduceMotion.matches?9:0, running=!reduceMotion.matches, lastPaint=0;
const smooth=(v,min,max)=>{const p=Math.min(1,Math.max(0,(v-min)/(max-min)));return p*p*(3-2*p)};
function motionLabel(){
  const next=running?['Pause Ⅱ','Pause reconstruction']:seconds>=13||reduceMotion.matches?['Replay ↻','Replay reconstruction']:['Play ▸','Resume reconstruction'];
  toggle.textContent=next[0];toggle.setAttribute('aria-label',next[1]);
}
function advance(now){
  if(!running)return;
  if(lastPaint)seconds=Math.min(13,seconds+(now-lastPaint)/1000);
  lastPaint=now;overview();
  if(seconds>=13){running=false;motionLabel();return}
  requestAnimationFrame(advance);
}
toggle.onclick=()=>{if(running){running=false}else{if(seconds>=13||reduceMotion.matches)seconds=0;running=true;lastPaint=0;requestAnimationFrame(advance)}motionLabel()};
document.addEventListener('visibilitychange',()=>{lastPaint=0});
reduceMotion.addEventListener('change',e=>{if(e.matches){running=false;seconds=9;overview();motionLabel()}});
motionLabel();if(running)requestAnimationFrame(advance);
function overview(){
  const {ctx,width,height}=surface(byId('overview')), left=2,right=width-2,base=height-22;
  const mx=w=>left+(w-data.x[0])/(data.x.at(-1)-data.x[0])*(right-left);
  const weights=[0,1,2,3].map(i=>smooth(seconds,i*1.65+.6,i*1.65+1.8));
  const stage=Math.min(4,Math.floor(Math.max(0,seconds-.6)/1.65)+Number(seconds>=.6));
  const full=weights[3]>=1, morph=smooth(seconds,9.15,11.35);
  const state=morph>.3?'FANTASY':full?'Full optical fit':['Broad Hβ','+ Narrow lines','+ Fe II','+ Host starlight','+ Remaining emission'][stage];
  if(ribbonStatus.textContent!==state)ribbonStatus.textContent=state;
  const fitted=data.x.map((w,i)=>data.hb[i]+weights[0]*data.narrow[i]+weights[1]*data.feii[i]+weights[2]*data.host[i]+weights[3]*(data.disk[i]+data.balmer[i]+data.broadOther[i]));
  const max=Math.max(...fitted), min=Math.min(...fitted), markHeight=base*.8,markWidth=markHeight*(mark.naturalWidth||1)/(mark.naturalHeight||1);
  const points=data.x.map((w,i)=>{const u=(w-data.x[0])/(data.x.at(-1)-data.x[0]), fraction=u*(crest.length-1), k=Math.floor(fraction);
    const yc=crest[k]+((crest[k+1]??crest[k])-crest[k])*(fraction-k);
    const x=mx(w),y=base-(fitted[i]-min)*(base-14)/(max-min);
    return [x*(1-morph)+(width/2+(u-.5)*markWidth)*morph,y*(1-morph)+(5+markHeight*yc)*morph]});
  const grad=ctx.createLinearGradient(0,0,width,0);
  ['#8cbde0','#b7a7da','#f1a9af','#f8c584'].forEach((c,i)=>grad.addColorStop(i/3,c));
  const fill=ctx.createLinearGradient(0,0,0,base);fill.addColorStop(0,'#bbadba44');fill.addColorStop(1,'#b2b4bc08');
  ctx.globalAlpha=1-smooth(morph,.72,1);
  trace(ctx,points,full?grad:returnColors[stage],1.55,base,fill);
  ctx.globalAlpha=smooth(morph,.55,1);
  if(mark.complete && mark.naturalWidth)ctx.drawImage(mark,width/2-markWidth/2,5,markWidth,markHeight);
  ctx.font='bold 12px Arial';ctx.fillStyle='#e9f0e9';ctx.textAlign='center';ctx.fillText('F A N T A S Y',width/2,base+11);
  ctx.globalAlpha=1-morph;
  ctx.strokeStyle='#425454';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(left,base+.5);ctx.lineTo(right,base+.5);ctx.stroke();
  labelAt(ctx,'3750 Å',left,base+15,'#a9bcbb','left');labelAt(ctx,'6850 Å',right,base+15,'#a9bcbb','right');
  for(const [w,name] of [[4863,'Hβ'],[5008,'[O III]'],[6565,'Hα']]){
    if((w===6565&&!full)||(w===5008&&weights[0]<.5))continue;
    const peak=Math.max(...fitted.filter((_,i)=>Math.abs(data.x[i]-w)<14));
    labelAt(ctx,name,mx(w),Math.max(11,base-(peak-min)*(base-14)/(max-min)-7),'#ecf1e9');
  }
  ctx.globalAlpha=1;
}
const indices=data.x.map((w,i)=>w>=4550 && w<=5150 ? i : -1).filter(i=>i>=0);
const profiles=[indices.map(i=>data.hb[i])];
for(const key of ['narrow','feii','host'])profiles.push(profiles.at(-1).map((v,n)=>v+data[key][indices[n]]));
profiles.push(indices.map(i=>data.total[i]));
let current=0, displayed=profiles[0].slice(), frame=0;
function detail(){
  const {ctx,width,height}=surface(byId('detail')), left=40,right=width-8,top=35,base=height-38;
  const max=Math.max(...displayed)*1.09, mx=w=>left+(w-4550)/600*(right-left), my=v=>base-v/max*(base-top);
  let step=max<60?10:max<260?50:100;
  ctx.lineWidth=1;ctx.strokeStyle='#d6dbd1';
  for(let v=0;v<=max;v+=step){const y=my(v);ctx.beginPath();ctx.moveTo(left,y);ctx.lineTo(right,y);ctx.stroke();labelAt(ctx,String(v),left-7,y+3,'#51645a','right')}
  for(let w=4600;w<=5100;w+=width<500?200:100)labelAt(ctx,String(w),mx(w),base+16,'#51645a');
  labelAt(ctx,'Rest-frame wavelength (Å)',(left+right)/2,height-3,'#51645a');
  labelAt(ctx,'Flux density (10⁻¹⁷ erg s⁻¹ cm⁻² Å⁻¹)',left,17,'#51645a','left');
  const points=indices.map((i,n)=>[mx(data.x[i]),my(displayed[n])]);
  trace(ctx,points,'#264237',1.9,base,stages[current][1]+'39');
}
function setStage(index){
  cancelAnimationFrame(frame);const from=displayed.slice(),to=profiles[index],start=performance.now();current=index;
  byId('stage-title').textContent=index===0?'Broad Hβ':index===4?'Full fitted spectrum':'Broad Hβ + '+stages.slice(1,index+1).map(s=>s[0]).join(' + ');
  byId('stage-description').textContent=stages[index][2];
  function update(now){const p=reduceMotion.matches?1:Math.min(1,(now-start)/480), eased=p*p*(3-2*p);
    displayed=to.map((v,i)=>from[i]+(v-from[i])*eased);detail();if(p<1)frame=requestAnimationFrame(update)}
  frame=requestAnimationFrame(update);
}
new ResizeObserver(overview).observe(byId('overview'));
new ResizeObserver(detail).observe(byId('detail'));
setStage(0);
const cinema=byId('cinema'),player=byId('player');
const films={short:['assets/reconstruction.mp4','Reconstructing NGC 2885','16 seconds · original score'],long:['assets/light-to-spectrum.mp4','From galaxy light to broad Hβ','2 min 39 s · original score']};
let playingFilm='short';
function showFilm(which){
  const [source,title,duration]=films[which];playingFilm=which;
  byId('film-title').textContent=title;byId('film-duration').textContent=duration;
  byId('swap-film').textContent=which==='short'?'Watch full sequence ↗':'Watch fast reconstruction ↗';
  player.src=source;if(!cinema.open)cinema.showModal();
  player.play().catch(()=>{}); // Native controls also work when the browser requires a second gesture.
}
document.querySelectorAll('[data-film]').forEach(link=>link.addEventListener('click',e=>{
  if(typeof cinema.showModal!=='function'||e.ctrlKey||e.metaKey||e.shiftKey||e.altKey)return;
  e.preventDefault();showFilm(link.dataset.film);
}));
byId('swap-film').onclick=()=>showFilm(playingFilm==='short'?'long':'short');
byId('close-film').onclick=()=>cinema.close();
cinema.addEventListener('close',()=>{player.pause();player.removeAttribute('src');player.load()});
cinema.addEventListener('click',e=>{if(e.target!==cinema)return;const r=cinema.getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)cinema.close()});
byId('copy').onclick=async()=>{
  try{await navigator.clipboard.writeText(byId('commands').textContent);byId('copy').textContent='Copied'}
  catch{const range=document.createRange();range.selectNodeContents(byId('commands'));const selection=getSelection();selection.removeAllRanges();selection.addRange(range);byId('copy').textContent='Selected'}
  setTimeout(()=>{byId('copy').textContent='Copy'},1800);
};
