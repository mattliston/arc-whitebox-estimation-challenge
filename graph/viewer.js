/* Offline graph navigation. Display parents do not add scientific dependencies. */
'use strict';
const ResearchMap = (() => {
  const COLORS = {bg:'#111315',panel:'#191c1f',ink:'#e6e6e3',muted:'#9da2a6',line:'#596168',accent:'#dca78c',solved:'#91b79b',closed:'#c58b91',open:'#8bb4ce',capacity:'#c9b379',blocked:'#999fa5'};
  const escape = s => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const titles = {
    P2R0:'Mean activations from network weights',P2R0A:'Evaluation contract',P2R0B:'Width and depth migration',
    P2R0C:'Estimator design and compute',P2R0D:'Gaussian, Hermite and Mehler approximations',
    P2R0C1:'Accuracy–compute trade-off',P2R0E:'Recursive high-order closure',P2R0F:'Fitted moment corrections',
    P2R0G:'Late-layer source reconstruction',P2R0H:'Pure-closure evaluation',
    P2R0G19:'Deterministic late-layer propagation',P2R0G19B:'Late-layer correction chains',
    P2R0G19B_CARRIER:'Cost of supplying corrections',P2R0G19B_MATH_ACCELERATION:'Mathematical acceleration',
    P2R0G19B_FULL_AQ_RAW_GAUSSIAN_K4_NOGO:'Finite-rank Gaussian closure',
    P2R0G19B_AQ_CARRIED_PROJECTION_Q211:'Carried AQ projection',
    P2R0G19B_AQ_AFFINE_H4_BAND_FORMULA:'Affine fourth-order correction',
    P2R0G19B_AQ_PRICE_Q211_PAIR_MINUS_TRACE:'Price identity for q211',
    P2R0G19B_PHYSICAL_H2H3_ENDPOINT_ADJOINT:'H2/H3 endpoint adjoint'
  };
  const guide=typeof GUIDE!=='undefined'?GUIDE:typeof module!=='undefined'?require('./guide.json'):{nodes:{},terms:[],statuses:{},topics:[]};
  const development=typeof RELEASE_PATH!=='undefined'?RELEASE_PATH:typeof module!=='undefined'?require('./release_path.json'):{nodes:[],edges:[],display_parents:{}};
  const title = n => guide.nodes[n.id]?.title || titles[n.id] || n.title;
  const question = n => guide.nodes[n.id]?.summary || '';
  function context(tree,id){return ancestry(tree,id).reverse().find(p=>guide.nodes[p]?.summary&&p!==id);}
  function excerpt(text,max=300){
    text=String(text||'').replace(/\s+/g,' ').trim();if(text.length<=max)return text;
    const prefix=text.slice(0,max),end=Math.max(prefix.lastIndexOf('. '),prefix.lastIndexOf('; '));
    return end>max*.45?prefix.slice(0,end+1):prefix.slice(0,prefix.lastIndexOf(' '))+'…';
  }
  const termsFor=n=>guide.terms.filter(t=>new RegExp(t.pattern,'i').test([n.title,n.claim,n.criterion].join(' '))).sort((a,b)=>Number(new RegExp(b.pattern,'i').test(n.title))-Number(new RegExp(a.pattern,'i').test(n.title))||Number(!!a.full_glossary)-Number(!!b.full_glossary));
  const status = n => ({open:'Open question',solved:'Resolved',closed_proof:'Ruled out by proof',closed_measured:'Closed by measurement',closed_cost:'Over compute budget',closed_this_form:'Failed in this form',capacity_only:'Potential only',blocked:'Blocked',adopted:'Implemented',released:'Designated estimator'})[n.status] || n.status.replaceAll('_',' ');
  const statusColor=n=>n.status.startsWith('closed_')?COLORS.closed:n.status==='capacity_only'?COLORS.capacity:n.status==='solved'||n.status==='adopted'||n.status==='released'?COLORS.solved:COLORS[n.status]||COLORS.blocked;
  function wrap(text,max=30,limit=2) {
    const lines=[];let line='';
    for(const token of String(text).split(/\s+/)) {
      const words=token.length>max?token.match(new RegExp(`.{1,${max}}`,'g')):[token];
      for(const word of words){if(line&&(line+' '+word).length>max){lines.push(line);line=word;}else line+=(line?' ':'')+word;}
    }
    if(line)lines.push(line);
    if(lines.length>limit)return [...lines.slice(0,limit-1),lines[limit-1].slice(0,max-1)+'…'];
    return lines;
  }
  const WIDTH=210,HEIGHT=80,H_GAP=76,V_GAP=94;
  const D=typeof module!=='undefined'?require('./vendor/d3.min.js'):globalThis.d3;
  function makeTree(nodes,edges=null){
    const byId=new Map(nodes.map(n=>[n.id,n])),children=new Map(nodes.map(n=>[n.id,[]]));
    for(const n of nodes)if(children.has(n.parent))children.get(n.parent).push(n.id);
    children.forEach(ids=>ids.sort((a,b)=>a.localeCompare(b,undefined,{numeric:true})));
    const sizes=new Map(),closed=new Map(),openCounts=new Map(),closedStatuses=new Set(['solved','closed_proof','closed_measured','closed_cost','closed_this_form','capacity_only']);
    function count(id){let size=1,isClosed=closedStatuses.has(byId.get(id).status),open=byId.get(id).status==='open'?1:0;for(const c of children.get(id)||[]){size+=count(c);open+=openCounts.get(c);isClosed=isClosed&&closed.get(c);}sizes.set(id,size);openCounts.set(id,open);closed.set(id,isClosed);return size;}
    for(const n of nodes)if(!byId.has(n.parent))count(n.id);
    const collapsible=new Set(nodes.filter(n=>closed.get(n.id)&&children.get(n.id).length&&!closed.get(n.parent)).map(n=>n.id));
    return {byId,children,sizes,closed,openCounts,collapsible,edgePairs:edges?new Set(edges.map(e=>e.from+'>'+e.to)):null};
  }
  function ancestry(tree,id){const path=[],seen=new Set();while(tree.byId.has(id)&&!seen.has(id)){seen.add(id);path.unshift(id);id=tree.byId.get(id).parent;}return path;}
  const descendants=(tree,id)=>(tree.sizes.get(id)||1)-1;
  /* Adapted from the original render_graph.py layoutOne: same D3 tree,
     210 × 80 nodes, 76 / 94 spacing, numeric sibling order and eight-column
     leaf packing. Packed leaves keep edges to their actual display parents. */
  function layout(tree,root,expanded=new Set(tree.byId.keys())){
    const included=new Set();function visit(id){included.add(id);if(expanded.has(id))for(const c of tree.children.get(id)||[])visit(c);}visit(root);
    const localChildren=new Map([...included].map(id=>[id,[]]));
    [...included].filter(id=>id!==root).forEach(id=>localChildren.get(tree.byId.get(id).parent).push(id));
    localChildren.forEach(ids=>ids.sort((a,b)=>a.localeCompare(b,undefined,{numeric:true})));
    const gridLeaves=new Set(),GRID_COLS=8,GRID_MIN=9;
    function datum(id){
      const kids=localChildren.get(id)||[],leafKids=kids.filter(k=>(localChildren.get(k)||[]).length===0);
      if(kids.length<GRID_MIN||leafKids.length<4)return {id,children:kids.map(datum)};
      const treeKids=kids.filter(k=>(localChildren.get(k)||[]).length>0),cols=Math.min(GRID_COLS,leafKids.length),chains=[...Array(cols)].map(()=>[]);
      leafKids.forEach((k,i)=>{chains[i%cols].push(k);gridLeaves.add(k);});
      const chain=(list,i)=>({id:list[i],children:i+1<list.length?[chain(list,i+1)]:[]});
      return {id,children:[...treeKids.map(datum),...chains.map(c=>chain(c,0))]};
    }
    const hierarchy=D.hierarchy(datum(root));
    D.tree().nodeSize([WIDTH+H_GAP,HEIGHT+V_GAP]).separation((a,b)=>a.parent===b.parent?1:1.2)(hierarchy);
    const raw=hierarchy.descendants(),minX=D.min(raw,d=>d.x);
    return new Map(raw.map(d=>[d.data.id,{x:d.x-minX+WIDTH/2+70,y:d.y+HEIGHT/2+70,width:WIDTH,height:HEIGHT,grid:gridLeaves.has(d.data.id)}]));
  }
  /* Pack complete sibling subtrees into balanced columns. Every child stays
     below its parent; selected nodes never change this layout. */
function compactLayout(tree,root,expanded=new Set(tree.byId.keys()),budget=15000){
 const memo=new Map();let calls=0;
 function plan(id,budget){
  const kids=expanded.has(id)?tree.children.get(id)||[]:[];
  const fork=kids.length>1&&development.nodes.some(n=>n.id===id)?kids.length*(WIDTH+32)+(kids.length-1)*H_GAP:WIDTH;
  budget=Math.max(fork,Math.floor(budget/50)*50);const key=id+'|'+budget;if(memo.has(key))return memo.get(key);calls++;
  if(!kids.length){const p={id,width:WIDTH,height:HEIGHT,children:[]};memo.set(key,p);return p;}
  if(kids.length===1){const child=plan(kids[0],budget),p={id,width:child.width,height:HEIGHT+52+child.height,unary:child};memo.set(key,p);return p;}
  let best;
  const maxCols=Math.min(5,kids.length,Math.max(1,Math.floor((budget+H_GAP)/(WIDTH+32+H_GAP))));
  for(let cols=1;cols<=maxCols;cols++){
   const columns=Array.from({length:cols},()=>({height:0,width:0,items:[],kids:[],weight:0}));
   for(const kid of [...kids].sort((a,b)=>tree.sizes.get(b)-tree.sizes.get(a))){const col=columns.reduce((a,b)=>a.weight<=b.weight?a:b);col.kids.push(kid);col.weight+=tree.sizes.get(kid);}
   const extra=Math.max(0,budget-(cols-1)*H_GAP-cols*(WIDTH+32)),total=columns.reduce((a,c)=>a+c.weight,0);
   for(const col of columns){const available=WIDTH+extra*col.weight/total;
    for(const p of col.kids.map(k=>plan(k,available)).sort((a,b)=>a.height-b.height||a.id.localeCompare(b.id,undefined,{numeric:true}))){col.items.push({plan:p,y:col.height});col.height+=p.height+52;col.width=Math.max(col.width,p.width);}
   }
   const width=Math.max(WIDTH,columns.reduce((v,c)=>v+c.width+32,0)+(cols-1)*H_GAP),height=HEIGHT+52+Math.max(...columns.map(c=>c.height))-52;
   const result={id,width,height,columns};if(!best||height<best.height||height===best.height&&width<best.width)best=result;
  }
  memo.set(key,best);return best;
 }
 const rootPlan=plan(root,budget),positions=new Map();
 function place(p,x,y,incoming){
  positions.set(p.id,{x:x+p.width/2,y:y+HEIGHT/2,width:WIDTH,height:HEIGHT,channel:incoming});
  if(p.unary){place(p.unary,x,y+HEIGHT+52,null);return;}
  let cx=x;
  for(const c of p.columns||[]){for(const v of c.items){const left=cx+32+(c.width-v.plan.width)/2,top=y+HEIGHT+52+v.y;place(v.plan,left,top,v.y===0?null:{x:cx+16,y:top-24,top:y+HEIGHT+24});}cx+=c.width+32+H_GAP;}
 }
 place(rootPlan,70,70,null);return positions;
}
  function combineTrees(tree,releaseTree){
    const parents=development.display_parents||{},nodes=[...tree.byId.values()].map(n=>parents[n.id]?{...n,parent:parents[n.id]}:n).concat([...releaseTree.byId.values()]);
    const pairs=new Set([...(tree.edgePairs||[]),...(releaseTree.edgePairs||[])]);
    for(const e of development.edges||[])pairs.add(e.from+'>'+e.to);
    return makeTree(nodes,[...pairs].map(pair=>{const [from,to]=pair.split('>');return {from,to};}));
  }
  function combinedLayout(tree,root,releaseTree,releaseRoot,expanded=new Set(tree.byId.keys())){
    const combined=combineTrees(tree,releaseTree),shown=new Set([...expanded,...releaseTree.byId.keys()]);
    return compactLayout(combined,root,shown);
  }
  function bounds(positions){const ps=[...positions.values()],x=Math.min(...ps.map(p=>Math.min(p.x-WIDTH/2,p.channel?.x??Infinity))),y=Math.min(...ps.map(p=>p.y-HEIGHT/2));return {x,y,width:Math.max(...ps.map(p=>p.x+WIDTH/2))-x,height:Math.max(...ps.map(p=>p.y+HEIGHT/2))-y};}
  function fit(positions,width,height,pad=42,max=4){const b=bounds(positions),px=Math.min(pad,width*.12),py=Math.min(pad,height*.12),k=Math.max(.001,Math.min(max,(width-2*px)/b.width,(height-2*py)/b.height));return {k,x:(width-b.width*k)/2-b.x*k,y:(height-b.height*k)/2-b.y*k};}
  const screen=(p,camera)=>({x:p.x*camera.k+camera.x,y:p.y*camera.k+camera.y});
  function keepAnchor(camera,before,after){return before&&after?{...camera,x:camera.x+(before.x-after.x)*camera.k,y:camera.y+(before.y-after.y)*camera.k}:{...camera};}
  function sourceURL(path,github,local=false){if(/^https?:\/\//.test(path))return path;const [p,...frag]=path.split('#');return (local?'../':github)+p.split('/').map(encodeURIComponent).join('/')+(frag.length?'#'+frag.join('#'):'');}
  function directoryURL(path,github,local=false){
    const clean=String(path||'').replace(/^\.\/?/,'').replace(/\/$/,'');
    return (local?'../':github.replace('/blob/','/tree/'))+clean.split('/').map(encodeURIComponent).join('/')+(clean?'/':'');
  }
  function curve(a,b){
    if(b.channel){
      const c=b.channel,points=[[a.x,a.y+HEIGHT/2],[a.x,c.top],[c.x,c.top],[c.x,c.y],[b.x,c.y],[b.x,b.y-HEIGHT/2]],path=D.path();path.moveTo(...points[0]);
      for(let i=1;i<points.length-1;i++){const prev=points[i-1],p=points[i],next=points[i+1],d1=Math.hypot(p[0]-prev[0],p[1]-prev[1]),d2=Math.hypot(next[0]-p[0],next[1]-p[1]),r=Math.min(7,d1/2,d2/2);if(!d1||!d2){path.lineTo(...p);continue;}path.lineTo(p[0]-(p[0]-prev[0])*r/d1,p[1]-(p[1]-prev[1])*r/d1);path.quadraticCurveTo(...p,p[0]+(next[0]-p[0])*r/d2,p[1]+(next[1]-p[1])*r/d2);}
      path.lineTo(...points.at(-1));return path.toString();
    }
    if(b.grid){const railY=a.y+HEIGHT/2+14,gx=b.x-WIDTH/2-14;return `M${a.x},${a.y+HEIGHT/2}V${railY}H${gx}V${b.y}H${b.x-WIDTH/2}`;}
    const start=a.y+HEIGHT/2,end=b.y-HEIGHT/2,mid=(start+end)/2;return `M${a.x},${start}C${a.x},${mid} ${b.x},${mid} ${b.x},${end}`;
  }
  function crossPath(a,b,ordinal,positions){
    const lane=ordinal%6,delta=b.y-a.y;
    if(Math.abs(delta)<1){const direction=Math.floor(ordinal/6)%2===0?-1:1,rail=a.y+direction*(HEIGHT/2+18+lane*8);return `M${a.x},${a.y+direction*HEIGHT/2}L${a.x},${rail}L${b.x},${rail}L${b.x},${b.y+direction*HEIGHT/2}`;}
    const direction=delta>0?1:-1,sy=a.y+direction*(HEIGHT/2+24+lane*3),ty=b.y-direction*(HEIGHT/2+24+lane*3),box=bounds(positions),left=box.x-24-lane*8,right=box.x+box.width+24+lane*8,rail=(a.x-left+b.x-left)<(right-a.x+right-b.x)?left:right;
    return `M${a.x},${a.y+direction*HEIGHT/2}L${a.x},${sy}L${rail},${sy}L${rail},${ty}L${b.x},${ty}L${b.x},${b.y-direction*HEIGHT/2}`;
  }
  function relations(tree,positions,edges,selected){return edges.filter(e=>tree.byId.get(e.to)?.parent!==e.from&&(e.from===selected||e.to===selected)&&positions.has(e.from)&&positions.has(e.to)).map((e,i)=>`<path d="${crossPath(positions.get(e.from),positions.get(e.to),i,positions)}" class="cross-link" marker-end="url(#cross-arrow)" fill="none" stroke="${COLORS.accent}" stroke-width="1.1" vector-effect="non-scaling-stroke"><title>${escape(e.type.replaceAll('_',' '))}</title></path>`).join('');}
  function draw(tree,positions,expanded,selected,href,reading=false,nodeHref=null){
    const route=new Set(ancestry(tree,selected));let edges='',nodes='';
    for(const [id,p] of positions){const n=tree.byId.get(id),parent=positions.get(n.parent),active=id===selected;
      if(parent&&(!tree.edgePairs||tree.edgePairs.has(n.parent+'>'+id)))edges+=`<path marker-end="url(#tree-arrow)" data-edge="${escape(id)}" d="${curve(parent,p)}" fill="none" vector-effect="non-scaling-stroke" stroke="${route.has(id)?COLORS.accent:statusColor(n)}" stroke-opacity="${route.has(id)?1:.55}" stroke-width="${route.has(id)?1.6:.8}"/>`;
      nodes+=`<g class="node${active?' selected':''}" data-node="${escape(id)}" transform="translate(${p.x-WIDTH/2},${p.y-HEIGHT/2})"><a data-select="${escape(id)}" href="${escape(nodeHref?nodeHref(n):href(n.source))}" target="_blank" rel="noopener" tabindex="${reading?'0':'-1'}" aria-label="${escape(title(n))}. ${escape(status(n))}. ${tree.children.get(id).length} subquestions. Read explanation"><title>${escape(n.title+' · '+status(n))}</title><rect class="node-box" width="210" height="80" rx="2" fill="${COLORS.panel}" stroke="${statusColor(n)}" stroke-width="${active?2.4:1}" vector-effect="non-scaling-stroke"/><text class="node-status" x="10" y="16" fill="${statusColor(n)}" font-size="10.5" font-family="system-ui,sans-serif">${escape(n.stage_label||status(n))}</text>`;
      const lines=wrap(title(n),28,3);
      nodes+=`<rect class="selection-ring" opacity="${active?1:0}" x="-5" y="-5" width="220" height="90" rx="4" fill="none" stroke="${COLORS.accent}" stroke-width="2.5" vector-effect="non-scaling-stroke"/><text class="node-title" x="10" y="${lines.length===1?48:lines.length===2?39:31}" fill="${COLORS.ink}" font-size="12.5" font-family="system-ui,sans-serif">${lines.map((s,i)=>`<tspan x="10" dy="${i?16:0}">${escape(s)}</tspan>`).join('')}</text></a>`;
      if(!reading&&tree.collapsible.has(id))nodes+=`<g class="collapse-badge" data-toggle="${escape(id)}" transform="translate(196,4)" role="button" tabindex="-1" aria-label="${expanded.has(id)?'Collapse':'Expand'} ${descendants(tree,id)} closed descendants" aria-expanded="${expanded.has(id)}"><title>${descendants(tree,id)} closed descendants · click to expand or collapse</title><rect x="-18" y="-11" width="36" height="22" rx="3" fill="${COLORS.panel}" stroke="${COLORS.muted}" stroke-width=".6"/><text text-anchor="middle" y="4" fill="${COLORS.muted}" font-size="11" font-family="system-ui,sans-serif">${expanded.has(id)?'−':'+'+descendants(tree,id)}</text></g>`;
      nodes+='</g>';
    }
    const defs=`<defs><marker id="tree-arrow" viewBox="0 -4 8 8" refX="8" refY="0" markerWidth="7" markerHeight="7" markerUnits="userSpaceOnUse" orient="auto"><path d="M0,-4L8,0L0,4" fill="${COLORS.line}"/></marker><marker id="cross-arrow" viewBox="0 -4 8 8" refX="8" refY="0" markerWidth="8" markerHeight="8" markerUnits="userSpaceOnUse" orient="auto"><path d="M0,-4L8,0L0,4" fill="${COLORS.accent}"/></marker></defs>`;
    return defs+edges+nodes;
  }
  function hit(positions,camera,x,y){const wx=(x-camera.x)/camera.k,wy=(y-camera.y)/camera.k;for(const [id,p] of positions)if(Math.abs(wx-p.x)<=WIDTH/2&&Math.abs(wy-p.y)<=HEIGHT/2)return id;return null;}
  return {WIDTH,HEIGHT,H_GAP,V_GAP,COLORS,guide,escape,title,question,context,excerpt,termsFor,status,statusColor,wrap,makeTree,ancestry,descendants,layout,compactLayout,combineTrees,combinedLayout,bounds,fit,screen,keepAnchor,sourceURL,directoryURL,curve,crossPath,relations,draw,hit};
})();
if(typeof module!=='undefined')module.exports=ResearchMap;
if(typeof document!=='undefined')(() => {
  const M=ResearchMap,$=id=>document.getElementById(id),E=M.escape;
  if(/^https?:$/.test(location.protocol)){ $('linkmode').hidden=true; document.querySelector('label[for="linkmode"]').hidden=true; }
  const tree=M.makeTree(DEPENDENCIES.nodes,DEPENDENCIES.edges),records=new Map(GRAPH.nodes.map(n=>[n.id,n]));
  const releaseTree=M.makeTree(RELEASE_PATH.nodes,RELEASE_PATH.edges);
  const combinedTree=M.combineTrees(tree,releaseTree);
  const developmentSteps=RELEASE_PATH.reading_order||[],developmentIds=new Set();
  for(const id of ['build-final-337619','build-final-337641'])M.ancestry(combinedTree,id).forEach(p=>developmentIds.add(p));
  const developmentEdges=new Map(RELEASE_PATH.edges.map(e=>[e.to,e]));
  const all=M.combinedLayout(tree,DEPENDENCIES.root,releaseTree,RELEASE_PATH.root),github=GRAPH.meta.github||'https://github.com/mattliston/arc-whitebox-estimation-challenge/blob/main/';
  const relatedMethods=new Map();
  for(const n of DEPENDENCIES.nodes)for(const id of n.related){if(!relatedMethods.has(id))relatedMethods.set(id,[]);relatedMethods.get(id).push(n.id);}
  const searchIndex=[...DEPENDENCIES.nodes,...GRAPH.nodes,...RELEASE_PATH.nodes].map(n=>({n,text:[n.id,n.title,M.title(n),M.question(n),n.original_label,n.status,n.claim,...(n.tags||[])].join(' ').toLowerCase(),title:(M.title(n)||n.title).toLowerCase()}));
  let expanded=new Set(tree.byId.keys()),selected=DEPENDENCIES.root,hover=null,focusRoot=DEPENDENCIES.root,mode='map';
  let position={x:0,y:0,k:1},locations=new Map(),offset=0,drag=null,ignoreClick=false,autoFit=true,navIndex=0,currentHash='';
  let searchMatches=[],activeResult=-1,guideOpen=false,animation=0,rendered={...position},miniCamera=null;
  const pointers=new Map(),activeStatuses=new Set(DEPENDENCIES.nodes.map(n=>n.status));
  const activeTree=()=>combinedTree;
  const getNode=id=>releaseTree.byId.get(id)||tree.byId.get(id)||records.get(id);
  const href=path=>M.sourceURL(path,github,$('linkmode').value==='local');
  const nodePath=n=>n.directory+'/'+(n.document||'README.md');
  const dirHref=path=>M.directoryURL(path,github,$('linkmode').value==='local');
  const workFor=id=>WORK_INDEX.nodes[id]||{primary:null,directories:[]};
  const nodeHref=n=>n.id===DEPENDENCIES.root?dirHref(''):workFor(n.id).primary?dirHref(workFor(n.id).primary):'#node='+encodeURIComponent(n.id);
  const reading=M.guide.reading,approaches=reading.approaches,approachFor=new Map(approaches.flatMap(a=>a.member_ids.map(id=>[id,a])));
  const shownChildren=new Set();
  const link=(path,label,cls='')=>`<a class="${cls}" href="${E(href(path))}" target="_blank" rel="noopener">${E(label)}</a>`;
  const jump=(id,label,cls='')=>`<a class="${cls}" href="#node=${encodeURIComponent(id)}" data-jump="${E(id)}">${E(label)}</a>`;
  const section=(title,body)=>'<h3>'+E(title)+'</h3>'+body;
  const announce=text=>{$('announce').textContent=text;};
  function filesHTML(files){return '<ul class="file-list">'+files.map(f=>{const parts=f.path.split('#')[0].split('/');return '<li>'+link(f.path,parts.at(-1))+'<span class="file-path">'+E(parts.slice(-3,-1).join('/'))+'</span></li>';}).join('')+'</ul>';}
  function filesSection(files,title){return files.length?section(title+' · '+files.length,filesHTML(files.slice(0,3)))+(files.length>3?`<details><summary>Show ${files.length-3} more files</summary>${filesHTML(files.slice(3))}</details>`:''):'';}
  function claimHTML(claim){const short=M.excerpt(claim,260);return '<p class="record-excerpt">'+E(short)+'</p>'+(short!==claim?'<details><summary>Full recorded result</summary><p>'+E(claim)+'</p></details>':'');}
  function termHTML(n){
    const terms=M.termsFor(n);if(!terms.length)return '';
    const list=ts=>'<dl class="terms">'+ts.map(t=>'<dt>'+link(t.source,t.label+' ↗')+'</dt><dd>'+E(t.full_glossary?M.excerpt(t.meaning,300):t.meaning)+(t.confidence==='inferred'?' <em>Interpretation inferred in the glossary.</em>':'')+'</dd>').join('')+'</dl>';
    return section('Terms used here',list(terms.slice(0,3)))+(terms.length>3?'<details><summary>'+ (terms.length-3)+' more definitions</summary>'+list(terms.slice(3))+'</details>':'');
  }
  function scopeNote(){return '<details class="scope-note"><summary>Source and scope</summary><p>This is the original Phase 2 problem map. Its metadata is dated '+E(DEPENDENCIES.meta.source_updated)+'. Status colors describe that record, including failed approaches; they are not a claim about the later released estimator.</p><p>Plain-language questions are reading aids. Exact research notes and all recorded dependencies remain linked.</p><p>'+link('graph/schema.md','Graph definitions ↗')+' · '+link('graph/GLOSSARY.md','Full glossary ↗')+'</p></details>';}
  function topicList(){return '<ul class="topic-list">'+M.guide.topics.map(id=>{const n=tree.byId.get(id);return '<li>'+jump(id,M.title(n))+'<small>'+tree.sizes.get(id)+' questions</small></li>';}).join('')+'</ul>';}
  function pathHTML(id){const path=M.ancestry(tree,id);return '<details class="location-trail"><summary>Location in the research tree</summary><ol>'+path.map(p=>'<li>'+jump(p,M.title(tree.byId.get(p)))+'</li>').join('')+'</ol></details>';}
  const plainStatus={
    open:'This question was still unresolved in the research record.',
    solved:'A reusable result or implementation was established for this question.',
    closed_proof:'A proof ruled out this approach under its stated assumptions.',
    closed_measured:'Measurements ruled out this version of the approach.',
    closed_cost:'This approach exceeded the allowed computation budget.',
    closed_this_form:'This version failed. Other versions were still possible.',
    capacity_only:'The idea showed promise with information the estimator could not yet supply.',
    blocked:'Progress depended on another unresolved question.'
  };
  function folderLink(path,label,cls=''){return '<a class="'+cls+'" href="'+E(dirHref(path))+'" target="_blank" rel="noopener">'+E(label)+'</a>';}
  function workActions(n){
    const w=workFor(n.id);
    if(!w.primary)return '<p class="reading-note">No separate work folder was recovered for this record.</p>';
    return '<div class="actions">'+folderLink(w.primary,'Open work folder ↗','primary')+(w.directories.length>1?'<button data-work="'+E(n.id)+'">All '+w.directories.length+' folders</button>':'')+'</div>';
  }
  function workFolders(id){const w=workFor(id);return w.directories.length?'<section class="work-folders">'+section('Code and results folders','<ul>'+w.directories.map(f=>'<li>'+folderLink(f.path,f.title)+'<span class="file-path">'+E(f.path)+'</span></li>').join('')+'</ul>')+'</section>':'';}
  function guideHTML(index=false){
    if(index)return '<h2>Approaches explored</h2><p class="reading-note">Choose an approach, then follow its individual attempts and results.</p><ol class="approach-list">'+approaches.map(a=>'<li>'+jump(a.id,a.title)+'<p>'+E(a.idea)+'</p></li>').join('')+'</ol><button class="text-button" data-guide>← Problem statement</button>';
    const p=reading.problem;
    return '<div class="eyebrow">The problem</div><h2>'+E(p.title)+'</h2><p class="lede">'+E(p.summary)+'</p><p class="reading-note">'+E(p.why_it_is_hard)+'</p><p class="reading-note">'+E(p.constraint)+'</p><div class="start-reading">'+jump('P2R0D','Start with the analytic baseline →','reading-start')+'<button class="text-button" data-approaches>Browse all six approaches</button></div><p class="reading-note">Each step explains the idea, its result, and the questions explored beneath it.</p><div class="setup-links">'+reading.problem.setup_ids.map(id=>jump(id,M.title(tree.byId.get(id)))).join('')+'</div>';
  }
  function readingNavigation(id){
    const a=approachFor.get(id),n=tree.byId.get(id),children=tree.children.get(id)||[];
    const preferred=a?.id===id?(a.key_nodes||[]).filter(c=>children.includes(c)):[];
    const ordered=[...new Set([...preferred,...children])],visible=shownChildren.has(id)?ordered:ordered.slice(0,3);
    let html='';
    if(ordered.length)html+='<section class="reading-children"><h3>Explore the attempts</h3><ul>'+visible.map(c=>{const child=tree.byId.get(c);return '<li style="--status:'+M.statusColor(child)+'">'+jump(c,M.title(child))+'<span>'+E(M.excerpt(M.question(child),105))+'</span></li>';}).join('')+'</ul>'+(ordered.length>visible.length?'<button class="text-button" data-children="'+E(id)+'">Show all '+ordered.length+' attempts</button>':'')+'</section>';
    if(a?.id===id){const next=approaches[approaches.indexOf(a)+1];html+='<nav class="reading-next" aria-label="Read another approach">'+(next?jump(next.id,'Next approach: '+next.title+' →'):'<button class="text-button" data-release>See the released estimators →</button>')+'</nav>';}
    else if(a)html+='<nav class="reading-next" aria-label="Return to approach">'+jump(a.id,'← '+a.title)+'</nav>';
    return html;
  }
  function developmentNavigation(id){
    if(mode!=='release'||!developmentIds.has(id)&&!releaseTree.byId.has(id))return '';
    const index=developmentSteps.indexOf(id),isFinal=id.startsWith('build-final-'),parent=combinedTree.byId.get(id)?.parent;
    const next=index>=0&&index<developmentSteps.length-1?[developmentSteps[index+1]]:combinedTree.children.get(id)?.filter(c=>releaseTree.byId.has(c)&&developmentIds.has(c))||[];
    const previous=index>0?developmentSteps[index-1]:parent;
    return '<nav class="development-nav" aria-label="Estimator development">'+(isFinal?'<span class="path-progress">Designated estimator</span>':index>=0?'<span class="path-progress">Development · '+(index+1)+' of '+developmentSteps.length+' shared milestones</span>':'<span class="path-progress">'+(developmentIds.has(id)?'Final variant':'Alternative tested')+'</span>')+
      next.map(c=>'<div class="development-next"><span>Next change</span>'+jump(c,M.title(getNode(c))+' →')+'</div>').join('')+
      (previous&&getNode(previous)?jump(previous,'← '+M.title(getNode(previous)),'development-previous'):'')+
      '<button class="text-button" data-release>Show the complete development path</button></nav>';
  }
  function developmentEvidence(id){
    const edge=developmentEdges.get(id);if(!edge)return '';
    return section(edge.type==='architecture_replacement'?'Why the core was rebuilt':'How this followed the previous work','<p>'+E(edge.explanation||'The build record identifies its input and output sources.')+'</p>')+(edge.sources||[]).map(f=>'<p>'+link(f.path,f.label)+'</p>').join('');
  }
  function technical(body){return '<details class="technical-details"><summary>Technical details</summary>'+body+'</details>';}
  function details(id,open=false){
    const n=getNode(id);if(!n)return;let html='';
    if(guideOpen||id===DEPENDENCIES.root){$('panel-label').textContent=guideOpen==='index'?'Approaches':'Start here';html=guideHTML(guideOpen==='index');}
    else if(releaseTree.byId.has(id)){
      $('panel-label').textContent='Estimator development';
      const edge=developmentEdges.get(id),label=edge?.type==='architecture_replacement'?'Core rebuilt':n.status==='closed_measured'?'Unsuccessful version':n.date?n.date.slice(0,10):n.step;
      html='<div class="eyebrow">'+E(label)+'</div><h2>'+E(n.title)+'</h2><p class="lede">'+E(n.claim)+'</p>'+workActions(n)+developmentNavigation(id);
      const attempts=(combinedTree.children.get(id)||[]).filter(c=>releaseTree.byId.has(c)&&!developmentIds.has(c));
      if(attempts.length)html+='<section class="reading-children"><h3>Other versions tested</h3><ul>'+attempts.map(c=>'<li style="--status:'+M.statusColor(getNode(c))+'">'+jump(c,M.title(getNode(c)))+'</li>').join('')+'</ul></section>';
      if(id.startsWith('build-final-'))html+='<p class="reading-note">'+folderLink('estimator','Reproduce this estimator ↗')+'</p>';
      let extra=developmentEvidence(id)+workFolders(id)+section('Exact source',link(n.source,'Open the exact source file ↗'))+filesSection(n.files,'Supporting files');
      if(n.record)extra+=section('Original record',jump(n.record,'Read the experiment or submission record'));
      extra+='<p>'+E(RELEASE_PATH.limit)+'</p>';
      html+=technical(extra);
    }else if(tree.byId.has(id)){
      $('panel-label').textContent=approachFor.get(id)?.title||'Research question';
      const editorial=M.guide.nodes[id],context=M.context(tree,id),code=n.files.filter(f=>/\.(py|sh)(?:#|$)/.test(f.path));
      html='<h2>'+E(M.title(n))+'</h2>';
      if(editorial?.summary)html+='<p class="lede">'+E(editorial.summary)+'</p>';
      else if(context)html+='<p class="context-line">Part of '+jump(context,M.title(tree.byId.get(context)))+'</p>';
      html+='<div class="outcome" style="--status:'+M.statusColor(n)+'"><strong>'+E(M.status(n))+'</strong><p>'+E(editorial?.plain_result||editorial?.result||plainStatus[n.status]||M.status(n))+'</p></div>';
      html+=workActions(n)+(mode==='release'&&developmentIds.has(id)?developmentNavigation(id):readingNavigation(id));
      let extra=developmentEvidence(id)+workFolders(id)+section('Original record','<p>'+E(n.title)+'</p><p>'+E(n.claim)+'</p>'+link(n.source,'Exact source record (JSON) ↗'))+section('Success criterion','<p>'+E(n.criterion||'No separate criterion was recorded.')+'</p>')+termHTML(n);
      if(n.files.length)extra+=filesSection(code,'Code')+filesSection(n.files.filter(f=>!code.includes(f)),'Evidence');
      const children=tree.children.get(id)||[];
      if(children.length)extra+=section('Subquestions','<ul>'+children.map(c=>'<li>'+jump(c,M.title(tree.byId.get(c)))+'</li>').join('')+'</ul>');
      const edges=DEPENDENCIES.edges.filter(e=>(e.from===id||e.to===id)&&!['decomposes_to','decomposition'].includes(e.type));
      if(edges.length)extra+=section('Other recorded connections','<ul>'+edges.map(e=>{const t=e.from===id?e.to:e.from;return '<li>'+jump(t,M.title(tree.byId.get(t)))+'<span class="relation-type">'+E(e.type.replaceAll('_',' '))+'</span></li>';}).join('')+'</ul>');
      if(n.related.length)extra+=section('Later records with the same artifacts','<ul>'+n.related.map(r=>'<li>'+jump(r,records.get(r)?.title||r)+'</li>').join('')+'</ul>');
      extra+='<p class="reading-note">Outcome recorded '+E(DEPENDENCIES.meta.source_updated)+'.</p><p>'+link('graph/GLOSSARY.md','Full glossary ↗')+' · '+link('graph/schema.md','Graph definitions ↗')+'</p><code class="node-id">'+E(id)+'</code>';
      html+=technical(extra);
    }else{
      $('panel-label').textContent='Research record';
      html='<div class="eyebrow">'+E(n.date.slice(0,10))+'</div><h2>'+E(n.title)+'</h2><p class="lede">'+E(M.excerpt(n.claim,220))+'</p>'+workActions(n);
      let extra=workFolders(id)+'<p>'+E(n.claim)+'</p>'+termHTML(n);
      if(n.measurement)extra+=section('Measurements','<div class="measurement">'+E(n.measurement)+'</div>');
      if(n.code_limitations)extra+='<p class="notice">'+E(n.code_limitations)+'</p>';
      extra+=filesSection(n.files,'Files')+filesSection(n.evidence.map(path=>({path})),'Evidence');
      const methods=relatedMethods.get(id)||[];
      if(methods.length)extra+=section('Related methods','<ul>'+methods.map(r=>'<li>'+jump(r,M.title(tree.byId.get(r)))+'</li>').join('')+'</ul>');
      const edges=GRAPH.edges.filter(e=>e.from===id||e.to===id);
      if(edges.length)extra+=section('Connected work','<ul>'+edges.map(e=>{const t=e.from===id?e.to:e.from;return '<li>'+jump(t,records.get(t).title)+'</li>';}).join('')+'</ul>');
      html+=technical(extra);
    }
    $('detail').innerHTML=html;$('detail').scrollTop=0;
    if(open)document.body.classList.add('detail-open');
  }
  function viewport(){const b=$('canvas').getBoundingClientRect();return {width:b.width,height:b.height};}
  function applyTransform(camera=position){rendered={...camera};const transform=`translate(${camera.x},${camera.y}) scale(${camera.k})`;$('drawing').setAttribute('transform',transform);$('relations').setAttribute('transform',transform);$('zoom-level').textContent=Math.round(camera.k*100)+'%';updateMinimap(camera);}
  function filters(){const counts=new Map();for(const n of DEPENDENCIES.nodes)counts.set(n.status,(counts.get(n.status)||0)+1);$('status-filters').innerHTML=[...counts].map(([s,n])=>`<label style="--status:${M.statusColor({status:s})}"><input type="checkbox" data-status="${E(s)}" ${activeStatuses.has(s)?'checked':''}>${E(M.status({status:s}))}<span>${n}</span></label>`).join('');}
  function applyFilter(){
    const keep=new Set(DEPENDENCIES.nodes.filter(n=>activeStatuses.has(n.status)).map(n=>n.id)),routes=new Set([...keep].flatMap(id=>M.ancestry(tree,id)));
    for(const button of document.querySelectorAll('[data-filter]')){const statuses=filterPreset(button.dataset.filter);button.setAttribute('aria-pressed',String(statuses.length===activeStatuses.size&&statuses.every(s=>activeStatuses.has(s))));}
    for(const group of $('drawing').querySelectorAll('[data-node]'))group.setAttribute('opacity',releaseTree.byId.has(group.dataset.node)||keep.has(group.dataset.node)||group.dataset.node===selected?'1':routes.has(group.dataset.node)?'.55':'.14');
    for(const edge of $('drawing').querySelectorAll('[data-edge]'))edge.setAttribute('opacity',releaseTree.byId.has(edge.dataset.edge)||routes.has(edge.dataset.edge)?'1':'.14');
    $('filter-note').hidden=activeStatuses.size===new Set(DEPENDENCIES.nodes.map(n=>n.status)).size;
  }
  function navigation(){
    const t=activeTree(),method=t.byId.has(selected),path=M.ancestry(t,method?selected:focusRoot),tail=path.slice(-3);
    $('breadcrumbs').innerHTML='<button data-guide>Problem</button>'+(path.length>3?'<span class="divider">/ …</span>':'')+tail.filter(id=>id!==DEPENDENCIES.root).map(id=>'<span class="divider">/</span>'+jump(id,M.title(t.byId.get(id)))).join('');
    $('recenter').disabled=!locations.has(selected);$('back').disabled=navIndex===0;$('parent').disabled=!method||!t.byId.has(t.byId.get(selected).parent);
    $('status-filter').hidden=false;$('status-legend').hidden=false;$('whole').setAttribute('aria-pressed',String(mode==='map'));$('released').setAttribute('aria-pressed',String(mode==='release'));
    $('view-caption').textContent=mode==='release'?'Estimator development · follow the highlighted changes from the analytic baseline':autoFit?'642 research questions · two designated estimators':M.title(getNode(selected))+' · full graph retained';
    $('view-help').textContent='Scroll to zoom · drag to pan · click a node for its result';
    $('minimap-wrap').hidden=autoFit;
  }
  function minimap(){
    miniCamera=M.fit(locations,180,120,7);const c=miniCamera;
    $('mini-drawing').innerHTML=[...locations].map(([id,p])=>'<rect x="'+(p.x-M.WIDTH/2)+'" y="'+(p.y-M.HEIGHT/2)+'" width="210" height="80" fill="'+M.statusColor(activeTree().byId.get(id))+'"/>').join('');
    $('mini-drawing').setAttribute('transform',`translate(${c.x},${c.y}) scale(${c.k})`);updateMinimap(position);
  }
  function updateMinimap(camera){
    if(!miniCamera)return;const b=viewport(),c=miniCamera,x=(0-camera.x)/camera.k,y=(0-camera.y)/camera.k;
    for(const [key,value] of Object.entries({x:x*c.k+c.x,y:y*c.k+c.y,width:b.width/camera.k*c.k,height:b.height/camera.k*c.k}))$('mini-viewport').setAttribute(key,value);
    const p=locations.get(selected);$('mini-selected').hidden=!p;if(p){$('mini-selected').setAttribute('cx',p.x*c.k+c.x);$('mini-selected').setAttribute('cy',p.y*c.k+c.y);}
  }
  function paint(anchor=null){
    const before=locations;
    locations=expanded.size===tree.byId.size?all:M.combinedLayout(tree,DEPENDENCIES.root,releaseTree,RELEASE_PATH.root,expanded);
    if(anchor)position=M.keepAnchor(position,before.get(anchor),locations.get(anchor));
    $('drawing').innerHTML=M.draw(combinedTree,locations,expanded,selected,href,false,nodeHref);
    updateSelection();minimap();applyTransform();navigation();
  }
  function moveCamera(target,animate=true){
    const token=++animation,from={...rendered},b=viewport();position={...target};autoFit=false;navigation();
    if(!animate||window.matchMedia?.('(prefers-reduced-motion: reduce)').matches){applyTransform();return;}
    const start=performance.now(),fromX=(b.width/2-from.x)/from.k,fromY=(b.height/2-from.y)/from.k,toX=(b.width/2-target.x)/target.k,toY=(b.height/2-target.y)/target.k;
    function frame(now){if(token!==animation)return;const t=Math.min(1,(now-start)/420),u=1-Math.pow(1-t,3),k=Math.exp(Math.log(from.k)+(Math.log(target.k)-Math.log(from.k))*u),cx=fromX+(toX-fromX)*u,cy=fromY+(toY-fromY)*u;applyTransform({k,x:b.width/2-cx*k,y:b.height/2-cy*k});if(t<1)requestAnimationFrame(frame);else applyTransform();}
    requestAnimationFrame(frame);
  }
  function fit(){animation++;const b=viewport();position=M.fit(locations,b.width,b.height,35,4);autoFit=true;applyTransform();navigation();}
  function closeSearch(clear=false){$('results').hidden=true;$('search').setAttribute('aria-expanded','false');$('search').removeAttribute('aria-activedescendant');activeResult=-1;if(clear)$('search').value='';$('clear-search').hidden=!$('search').value;}
  function closePanel(){document.body.classList.remove('detail-open');}
  function updateSelection(){
    const t=activeTree(),route=new Set(M.ancestry(t,hover||selected));
    for(const group of $('drawing').querySelectorAll('[data-node]')){const id=group.dataset.node,box=group.querySelector('.node-box');box.setAttribute('stroke',M.statusColor(t.byId.get(id)));box.setAttribute('stroke-width',id===hover?'1.8':'1');group.classList.toggle('selected',id===selected);group.querySelector('.selection-ring').setAttribute('opacity',id===selected?'1':'0');group.classList.toggle('release-highlight',mode==='release'&&developmentIds.has(id));}
    for(const edge of $('drawing').querySelectorAll('[data-edge]')){const active=route.has(edge.dataset.edge)||mode==='release'&&developmentIds.has(edge.dataset.edge);edge.setAttribute('stroke',active?M.COLORS.accent:M.statusColor(t.byId.get(edge.dataset.edge)));edge.setAttribute('stroke-width',active?'1.6':'.8');edge.setAttribute('stroke-opacity',active?'1':'.55');}
    $('relations').innerHTML=mode==='release'?'':M.relations(combinedTree,locations,DEPENDENCIES.edges.filter(e=>!['decomposes_to','decomposition'].includes(e.type)),selected);applyFilter();
  }
  function center(id,animate=true){const p=locations.get(id);if(!p)return;const b=viewport(),k=Math.max(.85,Math.min(1.1,position.k));moveCamera({k,x:b.width*.5-k*p.x,y:b.height*.43-k*p.y},animate);}
  function route(){return mode==='map'&&selected===DEPENDENCIES.root?'':`#node=${encodeURIComponent(selected)}&view=${mode}`;}
  function snapshot(){return {arcGraph:true,index:navIndex,selected,focusRoot,mode,position:{...position},expanded:[...expanded],statuses:[...activeStatuses],autoFit,guideOpen};}
  function save(){history.replaceState(snapshot(),'',location.pathname+location.search+route());currentHash=location.hash;}
  function navigate(action){save();action();navIndex++;history.pushState(snapshot(),'',location.pathname+location.search+route());currentHash=location.hash;navigation();}
  function select(id,{navigate:travel=false,push=true,open=true}={}){
    if(!getNode(id))return;
    const action=()=>{
      selected=id;guideOpen=false;hover=null;$('hover-info').hidden=true;
      if(releaseTree.byId.has(id)){mode='release';if(!locations.has(id)){M.ancestry(combinedTree,id).forEach(p=>{if(tree.byId.has(p))expanded.add(p);});paint();}updateSelection();if(travel||position.k<.45)center(id);}
      else if(tree.byId.has(id)){
        if(mode!=='map'&&!developmentIds.has(id)){mode='map';focusRoot=DEPENDENCIES.root;}
        if(!locations.has(id)){M.ancestry(combinedTree,id).forEach(p=>{if(tree.byId.has(p))expanded.add(p);});paint();}
        updateSelection();
        if(travel||position.k<.45)center(id);
      }else updateSelection();
      closeSearch(true);details(id,open);navigation();announce('Reading '+M.title(getNode(id)));
    };
    if(push&&(id!==selected||travel||guideOpen))navigate(action);else action();
  }
  function overview(){navigate(()=>{mode='map';focusRoot=DEPENDENCIES.root;expanded=new Set(tree.byId.keys());selected=DEPENDENCIES.root;guideOpen=false;hover=null;closeSearch(true);resetFilters();paint();fit();details(selected);closePanel();$('hover-info').hidden=true;});announce('Whole graph: 642 research questions and both released estimators.');}
  function filterPreset(kind){return kind==='closed'?['closed_proof','closed_measured','closed_cost','closed_this_form']:[kind];}
  function highlight(kind){const statuses=filterPreset(kind);if(activeStatuses.size===statuses.length&&statuses.every(s=>activeStatuses.has(s))){resetFilters();return;}activeStatuses.clear();statuses.forEach(s=>activeStatuses.add(s));filters();applyFilter();announce('Highlighted '+kind+' questions and their connecting paths.');}
  function resetFilters(){activeStatuses.clear();DEPENDENCIES.nodes.forEach(n=>activeStatuses.add(n.status));filters();applyFilter();}
  function showRelease(){navigate(()=>{mode='release';selected=RELEASE_PATH.path_start;guideOpen=false;hover=null;closeSearch(true);resetFilters();const missing=[...developmentIds].some(id=>!locations.has(id));if(missing){expanded=new Set(tree.byId.keys());paint();}else updateSelection();fit();details(selected,true);navigation();});announce('The highlighted path follows the estimator development from the analytic baseline through the recorded changes to both final versions.');}
  function showGuide(){navigate(()=>{selected=DEPENDENCIES.root;mode='map';guideOpen=false;hover=null;closeSearch(true);updateSelection();fit();details(selected,true);navigation();});}
  function locate(id){navigate(()=>{mode='map';focusRoot=DEPENDENCIES.root;expanded=new Set(tree.byId.keys());selected=id;guideOpen=false;paint();center(id);details(id);closePanel();});}
  function toggle(id){if(!tree.byId.has(id)||!locations.has(id)||!tree.children.get(id).length)return;navigate(()=>{if(expanded.has(id))expanded.delete(id);else expanded.add(id);selected=id;guideOpen=false;autoFit=false;paint(id);details(id,true);});}
  function searchResults(){
    const q=$('search').value.trim().toLowerCase();if(!q){closeSearch();return;}
    const terms=q.split(/\s+/);searchMatches=searchIndex.filter(r=>terms.every(t=>r.text.includes(t))).map(r=>({...r,rank:r.n.id.toLowerCase()===q?0:r.title.includes(q)?1:2})).sort((a,b)=>a.rank-b.rank);
    offset=Math.min(Math.max(0,offset),Math.max(0,Math.floor((searchMatches.length-1)/20)*20));activeResult=-1;
    const matches=searchMatches.slice(offset,offset+20);
    $('results').innerHTML='<div class="search-heading"><span>'+searchMatches.length.toLocaleString()+' results'+(matches.length?' · '+(offset+1)+'–'+(offset+matches.length):'')+'</span><button data-dismiss-search>Close</button></div><ol role="listbox" id="result-list" aria-label="Matching records">'+matches.map(({n},i)=>{
      const method=combinedTree.byId.has(n.id),parent=method?combinedTree.byId.get(combinedTree.byId.get(n.id).parent):null;
      return `<li role="option" id="search-result-${i}" aria-selected="false"><a class="result-link" href="#node=${encodeURIComponent(n.id)}" data-jump="${E(n.id)}">${E(M.title(n))}<span class="result-meta">${method?E(M.status(n))+(parent?' · '+E(M.title(parent)):''):E(n.kind)+' · '+E(n.date.slice(0,10))+' · '+E(n.outcome)}</span><span class="result-excerpt">${E(M.question(n)||M.excerpt(n.claim,130))}</span></a></li>`;
    }).join('')+'</ol>'+(matches.length?'':'<p class="empty">No matches. Try a topic, a term such as “covariance”, or a submission number.</p>')+(searchMatches.length>20?'<div class="pager"><button data-page="-1" '+(offset===0?'disabled':'')+'>Previous</button><button data-page="1" '+(offset+20>=searchMatches.length?'disabled':'')+'>Next</button></div>':'');
    $('results').hidden=false;$('clear-search').hidden=false;$('search').setAttribute('aria-expanded','true');$('search').removeAttribute('aria-activedescendant');announce(searchMatches.length+' search results');
  }
  function searchKey(event){
    if(event.key==='Escape'){closeSearch(true);$('search').blur();return;}
    if(!['ArrowDown','ArrowUp','Enter'].includes(event.key)||$('results').hidden)return;
    event.preventDefault();const count=Math.min(20,searchMatches.length-offset);if(!count)return;
    if(event.key==='Enter'){select(searchMatches[offset+Math.max(0,activeResult)].n.id,{navigate:true});return;}
    activeResult=Math.max(0,Math.min(count-1,activeResult+(event.key==='ArrowDown'?1:-1)));
    for(const el of $('results').querySelectorAll('[role="option"]'))el.setAttribute('aria-selected',String(el.id==='search-result-'+activeResult));
    $('search').setAttribute('aria-activedescendant','search-result-'+activeResult);$('search-result-'+activeResult)?.scrollIntoView({block:'nearest'});
  }
  function refreshLinks(){
    $('readme').href=href('README.md');$('reproduce').href=href('estimator/REPRODUCE.md');$('about').href=href('graph/schema.md');
    for(const a of $('drawing').querySelectorAll('[data-select]'))a.setAttribute('href',nodeHref(activeTree().byId.get(a.dataset.select)));
    details(selected);if(!$('results').hidden)searchResults();
  }
  document.addEventListener('click',event=>{
    if(event.ctrlKey||event.metaKey||event.shiftKey||event.altKey)return;
    for(const [attr,fn] of [['toggle',toggle],['read',id=>select(id,{navigate:true,open:false})],['locate',locate],['jump',id=>select(id,{navigate:true})]]){const t=event.target.closest('[data-'+attr+']');if(t){event.preventDefault();fn(t.dataset[attr]);return;}}
    const p=event.target.closest('[data-page]');if(p){offset+=Number(p.dataset.page)*20;searchResults();$('results').scrollTop=0;return;}
    const filter=event.target.closest('[data-filter]');if(filter)highlight(filter.dataset.filter);
    if(event.target.closest('[data-guide]'))showGuide();
    if(event.target.closest('[data-approaches]')){guideOpen='index';details(selected,true);}
    const childControl=event.target.closest('[data-children]');if(childControl){shownChildren.add(childControl.dataset.children);details(selected,true);}
    const workControl=event.target.closest('[data-work]');if(workControl){const fold=$('detail').querySelector('.technical-details');fold.open=true;fold.querySelector('.work-folders')?.scrollIntoView({block:'start',behavior:'smooth'});}
    if(event.target.closest('[data-release]'))showRelease();
    if(event.target.closest('[data-dismiss-search]'))closeSearch();
    if(event.target.closest('[data-reset-filters]'))resetFilters();
  });
  function eventNode(event){const a=event.target.closest('[data-select]');if(a)return a.dataset.select;const b=$('canvas').getBoundingClientRect();return M.hit(locations,position,event.clientX-b.left,event.clientY-b.top);}
  $('canvas').addEventListener('click',event=>{if(event.target.closest('[data-toggle],[data-jump]')||event.ctrlKey||event.metaKey||event.shiftKey||event.altKey)return;event.preventDefault();if(ignoreClick){ignoreClick=false;return;}const id=eventNode(event);if(id)select(id);else closePanel();});
  $('canvas').addEventListener('dblclick',event=>{event.preventDefault();const b=$('canvas').getBoundingClientRect();zoom(2,event.clientX-b.left,event.clientY-b.top);});
  $('search').addEventListener('input',()=>{offset=0;searchResults();});$('search').addEventListener('focus',()=>{if($('search').value.trim())searchResults();});$('search').addEventListener('keydown',searchKey);
  $('clear-search').addEventListener('click',()=>{closeSearch(true);$('search').focus();});
  document.addEventListener('change',event=>{const s=event.target.dataset?.status;if(s){if(event.target.checked)activeStatuses.add(s);else activeStatuses.delete(s);applyFilter();}});
  $('linkmode').addEventListener('change',refreshLinks);$('close-detail').addEventListener('click',closePanel);
  $('fit').addEventListener('click',fit);$('whole').addEventListener('click',overview);$('topics').addEventListener('click',showGuide);$('released').addEventListener('click',showRelease);
  $('back').addEventListener('click',()=>history.back());$('parent').addEventListener('click',()=>{const parent=activeTree().byId.get(selected)?.parent;if(parent)select(parent,{navigate:true,open:false});});
  $('recenter').addEventListener('click',()=>center(selected));
  function miniMove(event){const b=$('minimap').getBoundingClientRect(),x=(event.clientX-b.left)*180/b.width,y=(event.clientY-b.top)*120/b.height,worldX=(x-miniCamera.x)/miniCamera.k,worldY=(y-miniCamera.y)/miniCamera.k,v=viewport(),k=Math.max(.7,position.k);moveCamera({k,x:v.width/2-worldX*k,y:v.height/2-worldY*k},false);}
  $('minimap').addEventListener('pointerdown',event=>{event.preventDefault();$('minimap').setPointerCapture(event.pointerId);miniMove(event);});
  $('minimap').addEventListener('pointermove',event=>{if(event.buttons===1)miniMove(event);});
  function zoom(factor,x,y){animation++;position={...rendered};const k=Math.max(.001,Math.min(8,position.k*factor)),ratio=k/position.k;position={x:x-(x-position.x)*ratio,y:y-(y-position.y)*ratio,k};autoFit=false;applyTransform();navigation();}
  for(const [id,f] of [['minus',.8],['plus',1.25]])$(id).addEventListener('click',()=>{const b=viewport();zoom(f,b.width/2,b.height/2);});
  $('canvas').addEventListener('wheel',event=>{event.preventDefault();const b=$('canvas').getBoundingClientRect(),unit=event.deltaMode===1?16:event.deltaMode===2?b.height:1;zoom(Math.exp(-event.deltaY*unit*.002),event.clientX-b.left,event.clientY-b.top);},{passive:false});
  $('canvas').addEventListener('pointerdown',event=>{if(event.button!==undefined&&event.button!==0)return;animation++;position={...rendered};ignoreClick=false;pointers.set(event.pointerId,{x:event.clientX,y:event.clientY});drag={x:event.clientX,y:event.clientY,px:position.x,py:position.y,moved:false};});
  $('canvas').addEventListener('pointermove',event=>{
    if(!pointers.has(event.pointerId)){
      const id=eventNode(event);if(id!==hover){hover=id;updateSelection();$('hover-info').hidden=!id;if(id){const n=activeTree().byId.get(id);$('hover-info').innerHTML='<span style="color:'+M.statusColor(n)+'">'+E(M.status(n))+'</span><strong>'+E(M.title(n))+'</strong><p>'+E(M.question(n)||M.excerpt(n.claim,180))+'</p>';}}return;
    }
    const before=[...pointers.values()];pointers.set(event.pointerId,{x:event.clientX,y:event.clientY});
    if(pointers.size===2){const after=[...pointers.values()],b=$('canvas').getBoundingClientRect(),dist=ps=>Math.hypot(ps[0].x-ps[1].x,ps[0].y-ps[1].y),old=dist(before);if(old>0){const ox=(before[0].x+before[1].x)/2-b.left,oy=(before[0].y+before[1].y)/2-b.top,nx=(after[0].x+after[1].x)/2-b.left,ny=(after[0].y+after[1].y)/2-b.top;zoom(dist(after)/old,ox,oy);position.x+=nx-ox;position.y+=ny-oy;applyTransform();}ignoreClick=true;drag=null;return;}
    if(!drag||Math.hypot(event.clientX-drag.x,event.clientY-drag.y)<4&&!drag.moved)return;
    drag.moved=true;ignoreClick=true;autoFit=false;$('canvas').setPointerCapture(event.pointerId);$('canvas').classList.add('dragging');position.x=drag.px+event.clientX-drag.x;position.y=drag.py+event.clientY-drag.y;applyTransform();
  });
  function endPointer(event){pointers.delete(event.pointerId);drag=null;$('canvas').classList.remove('dragging');}
  $('canvas').addEventListener('pointerup',endPointer);$('canvas').addEventListener('pointercancel',endPointer);$('canvas').addEventListener('pointerleave',()=>{if(!pointers.size){hover=null;updateSelection();$('hover-info').hidden=true;}});
  $('canvas').addEventListener('keydown',event=>{
    const id=event.target.closest('[data-select]')?.dataset.select;
    if(event.key==='Enter'||event.key===' '){event.preventDefault();select(id||selected,{navigate:true});return;}
    const deltas={ArrowDown:[0,-70],ArrowUp:[0,70],ArrowLeft:[70,0],ArrowRight:[-70,0]};
    if(deltas[event.key]){event.preventDefault();position.x+=deltas[event.key][0];position.y+=deltas[event.key][1];autoFit=false;applyTransform();}
    if(event.key==='Home'){event.preventDefault();overview();}
    if(['+','=','-'].includes(event.key)){event.preventDefault();const b=viewport();zoom(event.key==='-'?.8:1.25,b.width/2,b.height/2);}
  });
  document.addEventListener('keydown',event=>{if(event.key==='Escape'){closeSearch();closePanel();$('canvas').focus();}if(event.key==='/'&&!['INPUT','TEXTAREA','SELECT'].includes(event.target.tagName)){event.preventDefault();$('search').focus();}});
  let resizeFrame;
  window.addEventListener('resize',()=>{cancelAnimationFrame(resizeFrame);resizeFrame=requestAnimationFrame(()=>{if(autoFit)fit();else{applyTransform();minimap();}});});
  function fromHash(){
    const params=new URLSearchParams(location.hash.slice(1)),id=params.get('node'),view=params.get('view');
    if(id&&getNode(id)){if(view==='map'&&tree.byId.has(id)){mode='map';selected=id;focusRoot=DEPENDENCIES.root;guideOpen=false;paint();fit();if(id!==DEPENDENCIES.root)center(id);details(id);}else {if(view==='release')mode='release';select(id,{navigate:true,push:false});}}
    else{mode='map';selected=DEPENDENCIES.root;focusRoot=DEPENDENCIES.root;paint();fit();details(selected);}
    currentHash=location.hash;
  }
  window.addEventListener('popstate',event=>{
    const s=event.state;if(!s?.arcGraph){fromHash();return;}
    navIndex=s.index;selected=s.selected;focusRoot=s.focusRoot;mode=s.mode==='release'?'release':'map';animation++;position={...s.position};expanded=new Set(s.expanded);activeStatuses.clear();s.statuses.forEach(v=>activeStatuses.add(v));autoFit=s.autoFit;guideOpen=s.guideOpen;hover=null;closeSearch(true);filters();paint();details(selected);closePanel();currentHash=location.hash;
  });
  window.addEventListener('hashchange',()=>{if(location.hash!==currentHash){fromHash();save();}});
  filters();paint();fit();fromHash();refreshLinks();save();
})();
