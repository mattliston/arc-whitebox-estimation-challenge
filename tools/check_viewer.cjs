#!/usr/bin/env node
/* Source targets, top-to-bottom geometry and interaction checks, without a browser. */
const fs=require('fs'),path=require('path'),assert=require('assert'),vm=require('vm');
const root=path.resolve(__dirname,'..'),M=require(path.join(root,'graph/viewer.js'));
const g=JSON.parse(fs.readFileSync(path.join(root,'graph/graph.json'))),d=JSON.parse(fs.readFileSync(path.join(root,'graph/dependency_map.json')));
const raw=JSON.parse(fs.readFileSync(path.join(root,'graph/dependency_source.json'))),tree=M.makeTree(d.nodes,d.edges);
const originalData=JSON.stringify(d),originalSources=JSON.stringify(raw);
const work=JSON.parse(fs.readFileSync(path.join(root,'graph/work_index.json'))),github='https://github.com/mattliston/arc-whitebox-estimation-challenge/blob/main/';
let checks=0,links=0,geometryCases=0;const check=(condition,message)=>{assert(condition,message);checks++;};
const lineCounts=new Map(),seenPaths=new Set();
function target(ref){
  if(/^https?:/.test(ref))return;
  const [rel,frag]=ref.split('#'),p=path.resolve(root,decodeURIComponent(rel));
  if(!seenPaths.has(p)){assert(p.startsWith(root+path.sep)&&fs.statSync(p).isFile(),'Missing/unsafe target: '+ref);seenPaths.add(p);}
  if(frag&&/^L\d+/.test(frag)){
    const m=frag.match(/^L(\d+)(?:-L?(\d+))?$/);
    if(m){if(!lineCounts.has(p))lineCounts.set(p,fs.readFileSync(p,'utf8').split('\n').length);assert(+m[1]>0&&+(m[2]||m[1])<=lineCounts.get(p),'Source line out of bounds: '+ref);}
  }
  links++;
}
check(d.nodes.length===642&&d.edges.length===791,'All Phase 2 nodes and original edges retained');
check(JSON.stringify(d.edges)===JSON.stringify(raw.edges),'Scientific edges preserved exactly');
const rawIds=new Set(raw.nodes.map(n=>n.id));
const rawById=new Map(raw.nodes.map(n=>[n.id,n]));
check(raw.nodes.every(n=>n.id.startsWith('P2R0')),'Phase 2 scope');
check(d.nodes.every(n=>rawIds.has(n.id)),'Every display node has an exact source');
for(const n of d.nodes){
  target(n.source);for(const f of n.files)target(f.path);check(M.ancestry(tree,n.id)[0]===d.root,'Node connected to original root: '+n.id);
  const original=rawById.get(n.id),record=JSON.parse(fs.readFileSync(path.join(root,n.source.split('#')[0]))).record;
  check(JSON.stringify(record)===JSON.stringify(original),'Every archived research object is preserved exactly: '+n.id);
  check(n.status===original.status&&n.claim===original.summary&&n.criterion===original.terminal_criterion&&n.original_label===original.label,'Research outcome and claims remain those of the original source: '+n.id);
}
for(const e of d.edges)check(rawIds.has(e.from)&&rawIds.has(e.to),'Scientific relationship endpoints exist');
for(const n of g.nodes){target(n.directory+'/'+(n.document||'README.md'));for(const f of n.files)target(f.path);for(const e of n.evidence)target(e);}
check(M.sourceURL('a b.py#L3','https://github.com/x/y/blob/main/')==='https://github.com/x/y/blob/main/a%20b.py#L3','GitHub source paths and line anchors');
check(M.sourceURL('a b.py#L3','unused',true)==='../a%20b.py#L3','Local source paths and anchors');
check(M.directoryURL('experiments/example work','https://github.com/x/y/blob/main/')==='https://github.com/x/y/tree/main/experiments/example%20work/','GitHub work links open directories with encoded path segments');
check(M.directoryURL('experiments/example work','unused',true)==='../experiments/example%20work/','Local work links open the existing directory');
check(M.directoryURL('',github)==='https://github.com/mattliston/arc-whitebox-estimation-challenge/tree/main/','Problem root opens the complete release repository');
const folderPaths=new Set();
for(const [id,entry] of Object.entries(work.nodes)){
  check(entry.primary===null||entry.directories.some(folder=>folder.path===entry.primary),'Primary work folder belongs to the node’s evidenced destinations: '+id);
  for(const folder of entry.directories){
    const p=path.resolve(root,folder.path);
    if(!folderPaths.has(p)){check((p===root||p.startsWith(root+path.sep))&&fs.statSync(p).isDirectory()&&!folder.path.endsWith('.json'),'Every work destination is an existing repository directory: '+folder.path);folderPaths.add(p);}
    check(Number.isInteger(folder.files)&&folder.files>=0&&Number.isInteger(folder.code_files)&&folder.code_files>=0,'Work counts distinguish cited files and existing Python programs');
  }
}
check(d.nodes.every(node=>work.nodes[node.id]),'Every original research node has an explicit work mapping');
check(g.nodes.every(node=>work.nodes[node.id]),'Every dated record has an explicit work mapping');
const missingWork=d.nodes.filter(node=>work.nodes[node.id].primary===null);
check(missingWork.length===21&&missingWork.every(node=>node.files.length===0&&work.nodes[node.id].availability==='unavailable'),'The 21 records without directly cited files never invent a work folder');

check(M.escape('<script>"&')==='&lt;script&gt;&quot;&amp;','Evidence labels escape HTML');

// Read the actual SVG route rather than reproducing the layout's routing rules.
function routePrimitives(svgPath){
  const tokens=svgPath.match(/[MLCQHV]|[-+]?(?:\d*\.\d+|\d+\.?\d*)(?:e[-+]?\d+)?/gi)||[],primitives=[];
  let at=0,point=null;
  const number=()=>Number(tokens[at++]),pair=()=>[number(),number()];
  while(at<tokens.length){
    const command=tokens[at++];
    if(command==='M'){point=pair();continue;}
    assert(point,'Route must begin with a move');
    let points;
    if(command==='L')points=[point,pair()];
    else if(command==='H')points=[point,[number(),point[1]]];
    else if(command==='V')points=[point,[point[0],number()]];
    else if(command==='Q')points=[point,pair(),pair()];
    else if(command==='C')points=[point,pair(),pair(),pair()];
    else assert.fail('Unrecognized connector command: '+command);
    assert(points.every(p=>p.every(Number.isFinite)),'Connector coordinates must be finite');
    primitives.push(points);point=points.at(-1);
  }
  return primitives;
}
function segmentIntersectsBox(a,b,box){
  let lo=0,hi=1;
  for(const [axis,min,max] of [[0,box.left,box.right],[1,box.top,box.bottom]]){
    const delta=b[axis]-a[axis];
    if(Math.abs(delta)<1e-12){if(a[axis]<min||a[axis]>max)return false;continue;}
    const t1=(min-a[axis])/delta,t2=(max-a[axis])/delta;
    lo=Math.max(lo,Math.min(t1,t2));hi=Math.min(hi,Math.max(t1,t2));
    if(lo>hi)return false;
  }
  return true;
}
function curveIntersectsBox(points,box,depth=0){
  const xs=points.map(p=>p[0]),ys=points.map(p=>p[1]),left=Math.min(...xs),right=Math.max(...xs),top=Math.min(...ys),bottom=Math.max(...ys);
  if(right<box.left||left>box.right||bottom<box.top||top>box.bottom)return false;
  if(points.length===2)return segmentIntersectsBox(points[0],points[1],box);
  if(depth===24||Math.max(right-left,bottom-top)<.001)return true;
  const levels=[points];while(levels.at(-1).length>1){const previous=levels.at(-1);levels.push(previous.slice(0,-1).map((p,i)=>[(p[0]+previous[i+1][0])/2,(p[1]+previous[i+1][1])/2]));}
  return curveIntersectsBox(levels.map(level=>level[0]),box,depth+1)||curveIntersectsBox(levels.map(level=>level.at(-1)).reverse(),box,depth+1);
}
function checkAllTreeRoutes(layoutTree,positions){
  let routes=0;const margin=1e-6,boxes=[...positions].map(([id,p])=>({id,left:p.x-M.WIDTH/2+margin,right:p.x+M.WIDTH/2-margin,top:p.y-M.HEIGHT/2+margin,bottom:p.y+M.HEIGHT/2-margin}));
  for(const [id,p] of positions){
    const parentId=layoutTree.byId.get(id).parent,parent=positions.get(parentId);
    if(!parent||layoutTree.edgePairs&&!layoutTree.edgePairs.has(parentId+'>'+id))continue;
    const primitives=routePrimitives(M.curve(parent,p));
    check(primitives.length>0,'Every drawn tree connector has a route: '+parentId+' → '+id);
    for(const box of boxes){if(box.id===id||box.id===parentId)continue;assert(!primitives.some(points=>curveIntersectsBox(points,box)),'Connector '+parentId+' → '+id+' crosses unrelated node '+box.id);}
    routes++;
  }
  checks++;return routes;
}

const legacy=M.layout(tree,d.root),expanded=new Set(rawIds),crypto=require('crypto');
const full=M.compactLayout(tree,d.root),legacyBounds=M.bounds(legacy),compactBounds=M.bounds(full);
check(full.size===642,'Compact overview retains every question');
check(compactBounds.width<legacyBounds.width*.3,'Overview is at least 70% narrower');
check(compactBounds.width/compactBounds.height>1.2&&compactBounds.width/compactBounds.height<1.8,'Overview has a balanced landscape aspect ratio');
for(const [id,p] of full){
  const parent=full.get(tree.byId.get(id).parent);if(parent)check(p.y>parent.y,'Every child remains below its actual parent');
  for(const [other,q] of full)if(other!==id)assert(!(Math.abs(p.x-q.x)<M.WIDTH&&Math.abs(p.y-q.y)<M.HEIGHT),'Node boxes never overlap');
  if(parent&&p.channel&&tree.edgePairs.has(tree.byId.get(id).parent+'>'+id)){
    const c=p.channel,points=[[parent.x,parent.y+M.HEIGHT/2],[parent.x,c.top],[c.x,c.top],[c.x,c.y],[p.x,c.y],[p.x,p.y-M.HEIGHT/2]];
    for(let i=1;i<points.length;i++)for(const [other,q] of full){
      if(other===id||other===tree.byId.get(id).parent)continue;
      const [a,b]=[points[i-1],points[i]],l=q.x-M.WIDTH/2,r=q.x+M.WIDTH/2,t=q.y-M.HEIGHT/2,bt=q.y+M.HEIGHT/2;
      const crosses=a[0]===b[0]?a[0]>l&&a[0]<r&&Math.max(a[1],b[1])>t&&Math.min(a[1],b[1])<bt:a[1]>t&&a[1]<bt&&Math.max(a[0],b[0])>l&&Math.min(a[0],b[0])<r;
      assert(!crosses,'Tree connector crosses unrelated node '+other);
    }
  }
}
checks++;
for(const id of tree.collapsible){const closed=new Set(expanded);closed.delete(id);const after=M.compactLayout(tree,d.root,closed);check(after.size===642-M.descendants(tree,id),'Collapse hides only the explicitly collapsed descendants');const c={x:12,y:36,k:.55},next=M.keepAnchor(c,full.get(id),after.get(id));check(Math.abs(M.screen(full.get(id),c).x-M.screen(after.get(id),next).x)<1e-8&&Math.abs(M.screen(full.get(id),c).y-M.screen(after.get(id),next).y)<1e-8,'Collapse preserves the selected anchor');}
for(const [width,height] of [[1040,700],[1440,760],[375,520]]){geometryCases++;const c=M.fit(full,width,height);check(compactBounds.width*c.k<=width&&compactBounds.height*c.k<=height,'Full overview fits the viewport');}
const href=p=>M.sourceURL(p,github);
const nodeHref=node=>node.id===d.root?M.directoryURL('',github):work.nodes[node.id]?.primary?M.directoryURL(work.nodes[node.id].primary,github):'index.html#node='+encodeURIComponent(node.id);
const markup=M.draw(tree,full,expanded,d.root,href,false,nodeHref);
check((markup.match(/data-node=/g)||[]).length===642,'All nodes rendered');
check((markup.match(/href=/g)||[]).length===642,'Every node links to work or its explicit unavailable-record explanation');
for(const node of d.nodes)check(markup.includes('data-select="'+M.escape(node.id)+'" href="'+M.escape(nodeHref(node))+'"'),'Rendered research node opens its evidenced work folder: '+node.id);
check(!/href="[^"]*dependency_records[^"]*\.json/.test(markup),'Primary graph hyperlinks do not open raw JSON records');
check((markup.match(/href="index\.html#node=/g)||[]).length===21,'Static links without a work folder open the interactive explanation');
check((markup.match(/class="collapse-badge"/g)||[]).length===29,'Original closed-subtree expansion controls');
check((markup.match(/data-edge=/g)||[]).length===614,'Only recorded tree relationships are drawn');
const sample=d.edges.find(e=>tree.byId.get(e.to).parent!==e.from);
check(M.relations(tree,full,d.edges,sample.from).includes('class="cross-link"'),'Non-tree relationships appear for the selected node');
check(!M.relations(tree,full,d.edges,null),'Non-tree relationships leave the overview unobstructed');
const template=fs.readFileSync(path.join(root,'graph/index.template.html'),'utf8');
check(template.includes('<h1>ARC White-Box Estimation Challenge - Phase 2</h1>'),'Requested page title');
check(template.includes('color-scheme:dark'),'Dark theme');
check(template.includes('/*__D3_JS__*/'),'Original D3 version embedded for offline use');
check(template.includes('id="minimap"'),'Clickable overview provides location context');
check(template.includes('@keyframes selection-pulse')&&template.includes('prefers-reduced-motion'),'Selected outline pulses and respects reduced motion');

const guide=JSON.parse(fs.readFileSync(path.join(root,'graph/guide.json'))),release=JSON.parse(fs.readFileSync(path.join(root,'graph/release_path.json')));
check(Object.keys(guide.nodes).length===642,'Every original research node has an editorial description');
for(const node of d.nodes){
  const entry=guide.nodes[node.id];
  check(entry&&typeof entry.title==='string'&&entry.title.trim().length>0&&typeof entry.summary==='string'&&entry.summary.trim().length>0,'Each research node has a readable title and explanation: '+node.id);
  if(node.id!==d.root)check(typeof entry.result==='string'&&entry.result.trim().length>0,'Each non-root research node states its recorded result: '+node.id);
  target(entry.source);check(entry.source===node.source,'Reading description links to its exact source');
}
const reading=guide.reading,memberIds=reading.approaches.flatMap(approach=>approach.member_ids),setupIds=reading.coverage.root_and_setup_ids;
check(reading.approaches.length===6,'Six approach groups provide the reading structure');
check(memberIds.length===638&&new Set(memberIds).size===638,'Approach groups partition 638 research nodes without duplication');
check(setupIds.length===4&&new Set(setupIds).size===4&&setupIds.includes(d.root),'Problem and setup contain the four ungrouped research nodes');
check(new Set([...memberIds,...setupIds]).size===642&&[...memberIds,...setupIds].every(id=>rawIds.has(id)),'Reading groups and setup cover exactly the original research graph');
check(reading.recommended_reading_order.length===reading.approaches.length&&reading.recommended_reading_order.every(id=>reading.approaches.some(approach=>approach.id===id)),'Recommended reading order contains every approach');
check(reading.reading_order_semantics.includes('Navigation')&&reading.reading_order_semantics.includes('only'),'Reading order is explicitly navigation rather than a new scientific dependency');
reading.problem.sources.forEach(target);
for(const approach of reading.approaches){
  check(approach.member_ids.includes(approach.id)&&approach.key_nodes.every(id=>approach.member_ids.includes(id)),'Approach entry and key nodes belong to their group');target(approach.source);
  check(rawIds.has(approach.next_id)||release.nodes.some(node=>node.id===approach.next_id),'Next reading destination is an existing research or release node');
  check(['recorded_connection','reading_navigation_only'].includes(approach.next_relation),'Next-step semantics distinguish reading navigation from recorded connections');
  if(approach.next_relation==='recorded_connection'){
    check(approach.next_recorded_edges?.length>0,'Recorded reading transition has explicit supporting edges');
    for(const edge of approach.next_recorded_edges)check(d.edges.some(original=>JSON.stringify(original)===JSON.stringify(edge)),'Reading transition cites an unchanged original edge');
  }else check(!approach.next_recorded_edges?.length,'Navigation-only transitions do not claim scientific edges');
}
check(JSON.stringify(d.edges)===JSON.stringify(raw.edges),'Reading navigation does not add or change any original directed edge');

for(const t of guide.terms){new RegExp(t.pattern,'i');target(t.source);}
check(guide.original_status_definitions.closed_measured.includes('failed'),'Failed accuracy status follows the original definition');
check(M.statusColor({status:'closed_measured'})!==M.statusColor({status:'solved'}),'Failures and resolved questions have different border colors');
check(M.statusColor({status:'open'})!==M.statusColor({status:'closed_proof'}),'Continuing and impossible routes have different border colors');
check(release.nodes.length===34,'The integrated development record contains all 34 added milestones and final build nodes');
for(const n of release.nodes){check(work.nodes[n.id]?.primary,'Every development and final build step opens its existing work folder: '+n.id);target(n.source);n.files.forEach(f=>target(f.path));if(n.record)check(g.nodes.some(r=>r.id===n.record),'Development step links to an existing research or evaluation record: '+n.id);}
for(const r of release.retained)target(r.source);
const releaseTree=M.makeTree(release.nodes,release.edges),releaseIds=new Set(release.nodes.map(n=>n.id));
const combinedTree=M.combineTrees(tree,releaseTree),expectedCount=d.nodes.length+release.nodes.length;
const combined=M.combinedLayout(tree,d.root,releaseTree,release.root),combinedBounds=M.bounds(combined),combinedExpanded=new Set([...expanded,...releaseIds]);
check(combined.size===expectedCount&&combinedTree.byId.size===expectedCount,'All research and development nodes share one integrated canvas');
for(const pair of [['build-export-337619','build-export-337641'],['build-final-337619','build-final-337641']]){const a=combined.get(pair[0]),b=combined.get(pair[1]);check(Math.abs(a.y-b.y)<1e-9&&Math.abs(a.x-b.x)>M.WIDTH,'The final variants appear as parallel branches, never as a serial chain');}
check(JSON.stringify(d)===originalData&&JSON.stringify(raw)===originalSources,'Integrated placement never mutates original research or source objects');
check(JSON.stringify([...tree.byId.values()])===JSON.stringify(d.nodes),'Original research tree nodes retain their exact display records');
check(Object.keys(release.display_parents).length===7,'Seven explicitly sourced display-parent changes integrate the early development');
check(JSON.stringify([...combined])===JSON.stringify([...M.compactLayout(combinedTree,d.root,combinedExpanded)]),'A single compact tree layout positions research and final development together');
for(const n of d.nodes){
  const displayed=combinedTree.byId.get(n.id),override=release.display_parents[n.id];
  check(JSON.stringify({...displayed,parent:n.parent})===JSON.stringify(n),'Integrated research nodes differ only in their declared display parent: '+n.id);
  check(displayed.parent===(override||n.parent),'Display parents change only where explicitly declared: '+n.id);
}
for(const [id,p] of combined){
  const parent=combined.get(combinedTree.byId.get(id).parent);if(parent)check(p.y-M.HEIGHT/2>parent.y+M.HEIGHT/2,'Every integrated child lies completely below its displayed parent: '+id);
  check(M.ancestry(combinedTree,id)[0]===d.root,'Every research and development node connects to the problem root: '+id);
  for(const [other,q] of combined)if(other!==id)assert(!(Math.abs(p.x-q.x)<M.WIDTH&&Math.abs(p.y-q.y)<M.HEIGHT),'Research and development boxes never overlap: '+id+' / '+other);
}
checks++;
const checkedIntegratedRoutes=checkAllTreeRoutes(combinedTree,combined);
check(!release.research_connections?.length,'Conceptual method associations do not stand in for development history');
const developmentPairs=new Map(release.edges.map(edge=>[edge.from+'>'+edge.to,edge]));
check(developmentPairs.size===release.edges.length,'Development connections are unique');
for(const edge of release.edges){
  check(combinedTree.byId.has(edge.from)&&combinedTree.byId.has(edge.to),'Development relationship endpoints exist: '+edge.from+' → '+edge.to);
  check(edge.type!=='method_association'&&typeof edge.explanation==='string'&&edge.explanation.trim().length>0&&edge.sources?.length>0,'Every development connection explains its meaning and cites archived evidence: '+edge.from+' → '+edge.to);
  edge.sources.forEach(source=>target(source.path));
  check(combinedTree.byId.get(edge.to).parent===edge.from,'Development edge is the displayed parent connection: '+edge.from+' → '+edge.to);
}
for(const id of [...releaseIds,...Object.keys(release.display_parents)]){
  const parent=combinedTree.byId.get(id).parent,edge=developmentPairs.get(parent+'>'+id);
  check(edge&&edge.sources?.length>0,'Every added or changed display-parent edge is sourced: '+parent+' → '+id);
}
for(const id of ['build-final-337619','build-final-337641']){
  const lineage=M.ancestry(combinedTree,id),required=[d.root,'P2R0D','P2R0E3','P2R0H7A','P2R0H7B','P2R0C1D14B','dev-recursive-correction','dev-signed-pair-state','build-parent','build-fit',id];
  check(lineage[0]===d.root,'Final estimator ancestry begins at the original problem: '+id);
  check(required.every((step,index)=>lineage.includes(step)&&(!index||lineage.indexOf(required[index-1])<lineage.indexOf(step))),'Final estimator follows the recorded baseline, corrections, rebuilt core and final fit: '+id);
  check(!lineage.includes('P2R0C'),'The final estimator does not appear to jump from the compute-cost topic: '+id);
  check(!lineage.some(step=>['dev-shared-basis-timeout','dev-second-fit-rejected'].includes(step)),'Failed branches do not enter final estimator ancestry: '+id);
  for(let i=1;i<lineage.length;i++)check(combinedTree.edgePairs.has(lineage[i-1]+'>'+lineage[i]),'Every connection on final ancestry is actually drawn: '+lineage[i-1]+' → '+lineage[i]);
}
check(release.reading_order.length===35&&new Set(release.reading_order).size===35,'The guided development sequence contains 35 distinct shared milestones');
check(release.reading_order[0]===release.path_start&&release.reading_order.at(-1)==='build-fit','The guided development starts at the analytic baseline and ends at the shared final fit');
for(let i=0;i<release.reading_order.length;i++){
  const id=release.reading_order[i];check(combinedTree.byId.has(id),'Reading milestone exists on the integrated map: '+id);
  if(i)check(combinedTree.byId.get(id).parent===release.reading_order[i-1],'Next reading milestone follows a displayed development edge: '+id);
}
check(!release.reading_order.some(id=>['dev-shared-basis-timeout','dev-second-fit-rejected'].includes(id)),'The guided successful development excludes the two failed side branches');
check(JSON.stringify(d.edges)===JSON.stringify(raw.edges),'Integrating development preserves all 791 original scientific dependencies');
const combinedMarkup=M.draw(combinedTree,combined,combinedExpanded,d.root,href,false,nodeHref);
check((combinedMarkup.match(/data-node=/g)||[]).length===expectedCount,'The shared canvas renders all research and development nodes');
check((combinedMarkup.match(/href=/g)||[]).length===expectedCount,'All shared-canvas nodes link to work or an unavailable-record explanation');
for(const node of [...d.nodes,...release.nodes])check(combinedMarkup.includes('data-select="'+M.escape(node.id)+'" href="'+M.escape(nodeHref(node))+'"'),'Shared-canvas hyperlink opens the expected work destination');
check(!/href="[^"]*dependency_records[^"]*\.json/.test(combinedMarkup),'Shared-canvas primary links never open source-record JSON');
check(!combinedMarkup.includes('stroke-dasharray'),'Integrated development uses solid parent connectors without dotted method links');
check((combinedMarkup.match(/data-edge=/g)||[]).length===checkedIntegratedRoutes,'Every tested integrated tree route is rendered exactly once');
check((combinedMarkup.match(/class="selection-ring" opacity="1"/g)||[]).length===1,'Static export shows exactly one selection ring');
check((combinedMarkup.match(/class="selection-ring" opacity="0"/g)||[]).length===expectedCount-1,'Unselected rings stay hidden without browser CSS');
for(const [width,height] of [[1040,700],[1440,980],[375,520]]){geometryCases++;const c=M.fit(combined,width,height);check(combinedBounds.width*c.k<=width&&combinedBounds.height*c.k<=height,'Combined overview fits the viewport');}
for(const [id,strength] of [['337619','1.5'],['337641','1.35']]){
  const exported=JSON.parse(fs.readFileSync(path.join(root,`release/${id}/seal_s${strength}/EXPORTED_v1.json`))),build=JSON.parse(fs.readFileSync(path.join(root,`release/${id}/build/BUILD_v1.json`)));
  const parentHash=crypto.createHash('sha256').update(fs.readFileSync(path.join(root,'submissions/337506/estimator.py'))).digest('hex');
  check(exported.files.release.parent_sha256===parentHash,'Readout export names the exact 337506 parent');
  check(exported.files.release.sha256===build.files.release.base_sha256,'Runtime build consumes the exact exported seal');
}

function preview(name,width,height,camera=null,selected=d.root){
  const top=132,bottom=42,c=camera||M.fit(combined,width,height-top-bottom),drawing=M.draw(combinedTree,combined,combinedExpanded,selected,href,false,nodeHref),transform=`translate(${c.x},${c.y+top}) scale(${c.k})`;
  const key=[['Resolved',M.COLORS.solved],['Closed',M.COLORS.closed],['Open',M.COLORS.open],['Potential only',M.COLORS.capacity],['Blocked',M.COLORS.blocked]].map(([label,color],i)=>`<rect x="${26+i*136}" y="81" width="11" height="11" rx="1" fill="${M.COLORS.panel}" stroke="${color}"/><text x="${44+i*136}" y="91" font-family="sans-serif" font-size="12" fill="${M.COLORS.muted}">${label}</text>`).join('');
  const svg=`<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}"><title>ARC White-Box Estimation Challenge - Phase 2</title><desc>All ${d.nodes.length} research questions and ${release.nodes.length} added development and build records form one integrated tree. The estimator development passes through the analytic baseline, recorded corrections, rebuilt core and final fitting. Colored borders show recorded outcomes. Open graph/index.html for interactive zoom and evidence.</desc><rect width="${width}" height="${height}" fill="${M.COLORS.bg}"/><g transform="${transform}">${drawing}</g><rect width="${width}" height="${top}" fill="${M.COLORS.bg}"/><text x="26" y="37" font-family="Georgia,serif" font-size="25" fill="${M.COLORS.ink}">ARC White-Box Estimation Challenge - Phase 2</text><text x="26" y="61" font-family="sans-serif" font-size="13" fill="${M.COLORS.muted}">${d.nodes.length} research questions · ${release.nodes.length} development and build records · linked code, experiments and results</text>${key}<path d="M752,87H780" stroke="${M.COLORS.line}" stroke-width="1.3"/><text x="789" y="91" font-family="sans-serif" font-size="12" fill="${M.COLORS.muted}">Recorded connection</text><text x="26" y="${height-16}" font-family="sans-serif" font-size="12" fill="${M.COLORS.muted}">Open graph/index.html to follow the estimator development and explore the surrounding research.</text></svg>`;
  check(!svg.includes('stroke-dasharray')&&!svg.includes('data-method-association'),'Static preview contains no misleading dotted method connections: '+name);
  fs.writeFileSync(path.join(root,name),svg);
}
preview('graph/dependencies.svg',1440,980);
const p=combined.get('P2R0F2');preview('_build/graph-detail.svg',1440,900,{k:1,x:650-p.x,y:150-p.y},'P2R0F2');
function developmentPreview(name,ids,selected){
  const nearby=new Map(ids.map(id=>[id,combined.get(id)]));
  check([...nearby.values()].every(Boolean),'Development preview contains existing graph nodes: '+name);
  preview(name,1440,1080,M.fit(nearby,1440,906,60,1.1),selected);
}
developmentPreview('_build/graph-core-rebuild.svg',['P2R0C1D14A','P2R0C1D14B','dev-recursive-correction','dev-signed-pair-state','dev-fitted-births','dev-composed-corrections','dev-shared-basis','dev-shared-basis-timeout'],'dev-signed-pair-state');
developmentPreview('_build/graph-final-fit.svg',['dev-compressed-messages','dev-edge-readout','dev-second-fit-rejected','build-parent','build-fit','build-export-337619','build-runtime-337619','build-final-337619','build-export-337641','build-runtime-337641','build-final-337641'],'build-fit');
fs.writeFileSync(path.join(root,'_build/viewer_validation.json'),JSON.stringify({
  checks,links,browser_rendered:false,phase2_nodes:d.nodes.length,phase2_edges:d.edges.length,
  original_source_objects_preserved:true,original_scientific_edges_preserved:true,original_layout_preserved:false,
  compact_layout_bounds:compactBounds,combined_layout_bounds:combinedBounds,combined_canvas_nodes:combined.size,
  integrated_development_nodes:release.nodes.length,source_backed_development_edges:release.edges.length,
  explicit_display_parent_overrides:Object.keys(release.display_parents).length,shared_development_milestones:release.reading_order.length,
  both_final_estimators_connected_to_problem:true,final_variants_drawn_in_parallel:true,final_paths_traverse_recorded_early_development:true,
  failed_attempts_are_side_branches:true,no_dotted_method_associations:true,
  static_preview_dimensions:[1440,980],development_detail_previews:['_build/graph-core-rebuild.svg','_build/graph-final-fit.svg'],static_selection_rings_correct:true,
  legacy_layout_bounds:legacyBounds,width_reduction:1-combinedBounds.width/legacyBounds.width,node_size:[210,80],closed_branch_badges:tree.collapsible.size,
  recorded_tree_connections:checkedIntegratedRoutes,contextual_cross_connections:true,layout_cases:geometryCases,top_to_bottom:true,nonoverlapping_nodes:true,
  all_integrated_tree_routes_checked:checkedIntegratedRoutes,tree_routes_avoid_unrelated_nodes:true,curve_check:'Actual SVG line, quadratic and cubic paths checked by recursive Bezier subdivision',
  expansion_keeps_anchor:true,status_colors_from_original_definitions:true,editorial_summaries:Object.keys(guide.nodes).length,editorial_results:641,
  reading_groups:reading.approaches.length,reading_coverage_nodes:memberIds.length+setupIds.length,reading_navigation_preserves_scientific_edges:true,
  work_index_nodes:Object.keys(work.nodes).length,verified_work_directories:folderPaths.size,primary_node_links_open_work_folders:true,explicit_missing_work_folders:missingWork.length,
  inline_definitions:guide.terms.length,verified_development_and_build_nodes:release.nodes.length,final_parent_hashes_verified:true,
  whole_graph_on_load:true,mobile_whole_graph:true,direct_source_links:true
},null,2)+'\n');
console.log(`${checks} checks and ${links} file links pass. All ${combined.size} nodes form one integrated map; ${checkedIntegratedRoutes} solid tree routes avoid unrelated boxes. Original research objects and source edges preserved; both final estimators connect through the recorded development to the problem. Browser rendering not tested.`);
