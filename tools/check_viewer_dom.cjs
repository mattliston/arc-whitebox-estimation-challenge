#!/usr/bin/env node
/* Interaction checks in jsdom, not a rendered browser.
   npm install --prefix /tmp/arc-graph-dom jsdom@27.0.1 --no-audit --no-fund
   NODE_PATH=/tmp/arc-graph-dom/node_modules node tools/check_viewer_dom.cjs */
'use strict';
const fs=require('fs'),path=require('path'),assert=require('assert');
const {JSDOM,VirtualConsole}=require('jsdom');
const root=path.resolve(__dirname,'..'),M=require(path.join(root,'graph/viewer.js'));
let checks=0;const check=(v,message)=>{assert(v,message);checks++;};
const d=JSON.parse(fs.readFileSync(path.join(root,'graph/dependency_map.json'))),tree=M.makeTree(d.nodes,d.edges);
const release=JSON.parse(fs.readFileSync(path.join(root,'graph/release_path.json'))),releaseTree=M.makeTree(release.nodes,release.edges),total=d.nodes.length+release.nodes.length;
const combinedTree=M.combineTrees(tree,releaseTree),full=M.combinedLayout(tree,d.root,releaseTree,release.root);
const developmentIds=new Set(['build-final-337619','build-final-337641'].flatMap(id=>M.ancestry(combinedTree,id)));
const work=JSON.parse(fs.readFileSync(path.join(root,'graph/work_index.json')));
const errors=[],vc=new VirtualConsole();vc.on('jsdomError',e=>errors.push(e));
const dom=new JSDOM(fs.readFileSync(path.join(root,'graph/index.html'),'utf8'),{url:'http://graph.local/graph/index.html',runScripts:'outside-only',pretendToBeVisual:true,virtualConsole:vc});
const w=dom.window,doc=w.document,$=id=>doc.getElementById(id);let size={width:1040,height:680};
w.SVGElement.prototype.getBoundingClientRect=()=>({...size,left:0,top:0,right:size.width,bottom:size.height});
w.SVGElement.prototype.setPointerCapture=()=>{};w.HTMLElement.prototype.scrollIntoView=function(){this.dataset.scrolled='true';};
let reduceMotion=true;w.matchMedia=()=>({matches:reduceMotion});
for(const script of doc.scripts)require('vm').runInContext(script.textContent,dom.getInternalVMContext());
const click=el=>{assert(el,'Click target exists');el.dispatchEvent(new w.MouseEvent('click',{bubbles:true,cancelable:true}));};
const key=(el,key)=>el.dispatchEvent(new w.KeyboardEvent('keydown',{key,bubbles:true,cancelable:true}));
const count=()=>$('drawing').querySelectorAll('[data-node]').length;
const has=id=>!!$('drawing').querySelector(`[data-node="${id}"]`);
const nodeState=()=>[...$('drawing').querySelectorAll('[data-node]')].map(node=>({node,id:node.dataset.node,transform:node.getAttribute('transform')}));
const sameNodes=before=>{const current=new Map([...$('drawing').querySelectorAll('[data-node]')].map(node=>[node.dataset.node,node]));return before.length===current.size&&before.every(({node,id,transform})=>node===current.get(id)&&node.getAttribute('transform')===transform);};
function checkConnections(stage){
  check(!doc.querySelector('#associations,.research-association'),`No detached method-association lines remain ${stage}`);
  check(!doc.querySelector('#map-labels,.map-landmark'),`No floating topic labels cover the map ${stage}`);
  for(const final of ['build-final-337619','build-final-337641']){
    const route=M.ancestry(combinedTree,final);
    check(route[0]===d.root&&route.includes('P2R0E3')&&route.includes('dev-signed-pair-state'),`The ${final} route passes through actual development ${stage}`);
    check(route.slice(1).every(id=>$('drawing').querySelector(`[data-edge="${id}"]`)),`Every step of ${final} has a drawn connection ${stage}`);
  }
}

function visiblePanel(){
  const copy=$('detail').cloneNode(true);
  for(const fold of [...copy.querySelectorAll('details:not([open])')].reverse())fold.replaceChildren(fold.querySelector('summary').cloneNode(true));
  for(const hidden of copy.querySelectorAll('[hidden]'))hidden.remove();
  return copy;
}
function inspectDisclosure(n){
  const folds=[...$('detail').children].filter(el=>el.tagName==='DETAILS');
  check(folds.length===1&&folds[0].classList.contains('technical-details'),'Inspector has one top-level technical disclosure');
  const fold=folds[0];
  check(!fold.open&&fold.querySelector('summary').textContent==='Technical details','Technical details start closed and have a clear label');
  check(!visiblePanel().querySelector('.file-list,.record-excerpt,.measurement,.terms'),'Files, measurements and terminology are hidden until requested');
  if(n&&n.claim.length>60){
    check(!visiblePanel().textContent.includes(n.claim),'The raw research note is absent from the initial inspector');
    fold.open=true;check(visiblePanel().textContent.includes(n.claim),'Opening Technical details makes the complete research note available');fold.open=false;
  }
}

const camera=()=>{const m=$('drawing').getAttribute('transform').match(/translate\(([-.\d]+),([-.\d]+)\) scale\(([-.\d]+)\)/);return {x:+m[1],y:+m[2],k:+m[3]};};
function search(q){$('search').value=q;$('search').dispatchEvent(new w.Event('input',{bubbles:true}));}
function jump(id){search(id);click($('results').querySelector(`[data-jump="${id}"]`));}
function nodeClick(id){click($('drawing').querySelector(`[data-select="${id}"]`));}
const back=()=>new Promise((resolve,reject)=>{const timer=setTimeout(()=>reject(new Error('Back navigation timed out')),1000);w.addEventListener('popstate',()=>{clearTimeout(timer);resolve();},{once:true});click($('back'));});
const forward=()=>new Promise((resolve,reject)=>{const timer=setTimeout(()=>reject(new Error('Forward navigation timed out')),1000);w.addEventListener('popstate',()=>{clearTimeout(timer);resolve();},{once:true});w.history.forward();});
function pointer(type,props){const e=new w.Event(type,{bubbles:true});Object.assign(e,props);$('canvas').dispatchEvent(e);}
function saveSVG(name){fs.writeFileSync(path.join(root,'_build',name),`<svg xmlns="http://www.w3.org/2000/svg" width="${size.width}" height="${size.height}" viewBox="0 0 ${size.width} ${size.height}"><rect width="100%" height="100%" fill="${M.COLORS.bg}"/>${$('canvas').innerHTML}</svg>`);}
(async()=>{
  check(errors.length===0,'Embedded HTML executes without errors');
  check(total===676&&count()===total,'The default view contains all 642 research nodes and 34 development/build records');
  checkConnections('on startup');
  check($('detail').querySelector('h2').textContent===M.guide.reading.problem.title,'Opening panel starts with the problem statement');
  check(!$('detail').querySelector('.best-public,.file-list,.measurement,details'),'Opening panel has no score box, file inventory or technical folds');
  check(visiblePanel().textContent.includes(M.guide.reading.problem.summary),'Opening panel states what the estimator must predict');
  check(!$('detail').textContent.includes('What are we trying'),'Tutorial questions are removed');
  const failure=d.nodes.find(n=>n.status==='closed_measured'),solved=d.nodes.find(n=>n.status==='solved');
  check($('drawing').querySelector(`[data-node="${failure.id}"] .node-box`).getAttribute('stroke')===M.COLORS.closed,'Failed routes retain their distinct border');
  check($('drawing').querySelector(`[data-node="${solved.id}"] .node-box`).getAttribute('stroke')===M.COLORS.solved,'Resolved routes retain their distinct border');
  click(doc.querySelector('[data-filter="open"]'));
  check(!$('filter-note').hidden&&count()===total,'Color highlighting preserves all questions');
  click(doc.querySelector('[data-filter="open"]'));check($('filter-note').hidden,'The same color restores all statuses');
  check([...$('drawing').querySelectorAll('[data-select]')].every(a=>!a.getAttribute('href').includes('dependency_records/')&&!/\.json(?:[?#]|$)/.test(a.getAttribute('href'))),'Graph node links open work locations rather than raw JSON records');
  const beforeReading=nodeState();
  click($('detail').querySelector('.reading-start[data-jump="P2R0D"]'));
  check(count()===total&&sameNodes(beforeReading)&&$('detail').querySelector('h2').textContent===M.title(tree.byId.get('P2R0D')),'Start reading opens the analytic baseline without replacing the graph');
  const primary=$('detail').querySelector('.actions a.primary');
  check(primary&&primary.getAttribute('href').includes('/tree/main/')&&!/\.json(?:[?#]|$)/.test(primary.getAttribute('href'))&&fs.statSync(path.join(root,work.nodes.P2R0D.primary)).isDirectory(),'The primary action opens the recovered code and results directory');
  for(const id of ['P2R0E','P2R0F']){
    click($('detail').querySelector(`.reading-next [data-jump="${id}"]`));
    check(count()===total&&sameNodes(beforeReading)&&$('detail').querySelector('h2').textContent===M.title(tree.byId.get(id)),`The reading sequence reaches ${id} while retaining the whole map`);
  }
  check($('detail').querySelectorAll('.reading-children [data-jump]').length<=3,'The reader introduces at most three attempts at once');
  click($('detail').querySelector('.reading-children [data-jump="P2R0F1"]'));
  check($('detail').querySelector('h2').textContent===M.title(tree.byId.get('P2R0F1'))&&count()===total,'An individual attempt opens in context');
  check(!$('parent').disabled&&$('detail').querySelector('.reading-next [data-jump="P2R0F"]'),'An attempt offers both parent navigation and a return to its approach');
  click($('detail').querySelector('.reading-next [data-jump="P2R0F"]'));
  check($('detail').querySelector('h2').textContent===M.title(tree.byId.get('P2R0F'))&&sameNodes(beforeReading),'Return to approach preserves every node position');
  click($('topics'));check($('detail').querySelector('.reading-start')&&sameNodes(beforeReading),'Start here restores the problem while retaining the complete map');const indexCamera=camera();click($('detail').querySelector('[data-approaches]'));
  check($('detail').querySelectorAll('.approach-list [data-jump]').length===6&&JSON.stringify(camera())===JSON.stringify(indexCamera),'The six-approach index opens without moving the camera');
  click($('detail').querySelector('.approach-list [data-jump="P2R0G"]'));
  check($('detail').querySelector('h2').textContent===M.title(tree.byId.get('P2R0G'))&&count()===total&&sameNodes(beforeReading),'The approach index opens another section without isolating it');
  const beforeChildren=camera();click($('detail').querySelector('[data-children="P2R0G"]'));
  check($('detail').querySelectorAll('.reading-children [data-jump]').length===tree.children.get('P2R0G').length&&JSON.stringify(camera())===JSON.stringify(beforeChildren)&&sameNodes(beforeReading),'Show all attempts expands only the reader and preserves the map');
  click($('whole'));
  const beforeNodes=nodeState();
  nodeClick('P2R0D');
  check(count()===total,'Selecting a node never isolates its descendants');
  check(sameNodes(beforeNodes),'Selection preserves every node object and world position');
  checkConnections('after ordinary node selection');
  check(camera().k>=.85,'Overview selection zooms to a readable surrounding section');
  check($('drawing').querySelector('[data-node="P2R0D"]').classList.contains('selected'),'Selection receives the pulse class');
  check($('drawing').querySelector('[data-node="P2R0D"] .selection-ring'),'Pulse is separate from the outcome-colored border');
  check(!$('minimap-wrap').hidden,'Overview locator appears when zoomed in');
  check($('mini-drawing').querySelectorAll('rect').length===total,'Locator represents the complete graph');
  check(visiblePanel().querySelector('.outcome p').textContent===(M.guide.nodes.P2R0D.plain_result||M.guide.nodes.P2R0D.result),'Selected panel gives the source-backed outcome in a short explanation');
  inspectDisclosure(tree.byId.get('P2R0D'));
  check($('breadcrumbs').textContent.includes('Gaussian'),'Breadcrumbs name the selected section');
  const heldCamera=camera();nodeClick('P2R0E');check(JSON.stringify(camera())===JSON.stringify(heldCamera),'A click while zoomed in keeps the camera stable');
  const beforePan=camera();pointer('pointerdown',{pointerId:1,clientX:400,clientY:300,button:0});pointer('pointermove',{pointerId:1,clientX:440,clientY:335});pointer('pointerup',{pointerId:1});
  check(camera().x!==beforePan.x,'Drag pans the continuous graph');
  const detail=$('detail').textContent;nodeClick('P2R0D');check($('detail').textContent===detail,'Drag does not accidentally select');
  const beforeLeave=camera();jump('P2R0G');check(count()===total,'Search navigation retains the whole graph');
  await back();check(JSON.stringify(camera())===JSON.stringify(beforeLeave),'Back restores the exact previous panned view');
  await forward();check($('detail').querySelector('h2').textContent===M.title(tree.byId.get('P2R0G')),'Forward restores the selected record');
  click($('parent'));check(count()===total&&$('detail').querySelector('h2').textContent===M.title(tree.byId.get('P2R0C')),'Parent navigation preserves the whole map');
  const current=camera();search('337641');check($('results').querySelector('.result-excerpt'),'Search includes result context');
  key($('search'),'ArrowDown');check($('search').getAttribute('aria-activedescendant')==='search-result-0','Keyboard search announces a selected result');
  key($('search'),'Enter');check($('detail').textContent.includes('337641'),'Enter opens the designated submission');
  check(JSON.stringify(camera())===JSON.stringify(current)&&count()===total,'A dated record does not replace the graph');
  $('linkmode').value='local';$('linkmode').dispatchEvent(new w.Event('change',{bubbles:true}));
  check($('detail').querySelector('.actions a').getAttribute('href')==='../submissions/337641/','File links follow local mode');
  inspectDisclosure();
  const beforeRelease=nodeState();
  click($('released'));
  check(count()===total&&has('build-final-337619')&&has('build-final-337641'),'Estimator development keeps both final sources and all research nodes in the same map');
  check(sameNodes(beforeRelease),'Development highlighting does not replace or move any map node');
  check([...full].every(([id,p])=>{const q=M.screen(p,camera());return q.x>=0&&q.x<=size.width&&q.y>=0&&q.y<=size.height;}),'Development highlighting begins with the complete graph in view');
  check($('drawing').querySelectorAll('.release-highlight').length===developmentIds.size,'The complete route through the research and development is highlighted');
  checkConnections('with estimator development highlighted');
  check($('detail').querySelector('h2').textContent===M.title(tree.byId.get(release.path_start)),'Development begins at the analytic baseline, not the release packaging');
  check(!$('drawing').querySelector('[data-node="dev-second-fit-rejected"]').classList.contains('release-highlight'),'A rejected parallel readout does not enter the successful route');
  saveSVG('graph-release-path.svg');
  for(let i=1;i<release.reading_order.length;i++){
    const id=release.reading_order[i],link=$('detail').querySelector(`.development-next [data-jump="${id}"]`);
    check(link,`The reader offers the recorded next milestone ${id}`);click(link);
    check(sameNodes(beforeRelease)&&$('detail').querySelector('h2').textContent===M.title(combinedTree.byId.get(id)),`Reading ${id} keeps the entire graph and its positions`);
    if(id==='dev-signed-pair-state'){
      check(visiblePanel().textContent.includes('independently rebuilt')&&$('detail').querySelector('.eyebrow').textContent==='Core rebuilt','The architecture replacement is explicit in the reader');
      saveSVG('graph-development-rebuild.svg');
    }
  }
  check(['build-export-337619','build-export-337641'].every(id=>$('detail').querySelector(`.development-next [data-jump="${id}"]`)),'The shared readout branches into both final variants');
  click($('detail').querySelector('.development-next [data-jump="build-export-337641"]'));
  check($('detail').querySelector('.development-next [data-jump="build-runtime-337641"]'),'Each exported variant leads to its own runtime assembly');
  inspectDisclosure();
  const beforeFinal=nodeState();nodeClick('build-final-337641');
  check(count()===total&&sameNodes(beforeFinal),'Selecting a final estimator preserves the complete map and node positions');
  check(visiblePanel().textContent.includes('1.4006490071107744e-9'),'Final node has the exact recorded score');inspectDisclosure();
  click($('whole'));check(count()===total,'Whole graph restores every research question and final-build step');
  checkConnections('after Whole graph');
  const leaf=tree.byId.get('P2R0G19B_MECH_N0_1_OPEN_PROBLEM');nodeClick(leaf.id);inspectDisclosure(leaf);
  check(visiblePanel().querySelector('.lede').textContent===M.guide.nodes[leaf.id].summary,'Deep questions also have a short explanation before the raw research note');
  click($('whole'));
  const closed=[...combinedTree.collapsible].find(id=>tree.children.get(id)?.length&&$('drawing').querySelector(`[data-toggle="${id}"]`)),oldCamera=camera(),anchor=M.screen(full.get(closed),oldCamera);
  click($('drawing').querySelector(`[data-toggle="${closed}"]`));check(count()===total-M.descendants(combinedTree,closed),'Only an explicit collapse control hides descendants');
  const expanded=new Set(tree.byId.keys());expanded.delete(closed);const collapsed=M.combinedLayout(tree,d.root,releaseTree,release.root,expanded),nextAnchor=M.screen(collapsed.get(closed),camera());
  check(Math.abs(anchor.x-nextAnchor.x)<1e-8&&Math.abs(anchor.y-nextAnchor.y)<1e-8,'Explicit collapse stays anchored');
  click($('drawing').querySelector(`[data-toggle="${closed}"]`));check(count()===total,'Explicit expansion restores all questions');
  click($('whole'));saveSVG('graph-compact-overview.svg');
  jump('P2R0F2');saveSVG('graph-context-selection.svg');
  const old=camera(),world={x:(400-old.x)/old.k,y:(300-old.y)/old.k};
  $('canvas').dispatchEvent(new w.WheelEvent('wheel',{deltaY:-100,clientX:400,clientY:300,bubbles:true,cancelable:true}));
  check(camera().k>old.k&&Math.abs(world.x*camera().k+camera().x-400)<1e-7&&Math.abs(world.y*camera().k+camera().y-300)<1e-7,'Wheel zoom stays under the pointer');
  const miniBefore=camera();$('minimap').dispatchEvent(Object.assign(new w.Event('pointerdown',{bubbles:true,cancelable:true}),{pointerId:2,clientX:250,clientY:200}));
  check(JSON.stringify(camera())!==JSON.stringify(miniBefore)&&count()===total,'Clicking the locator moves the camera without replacing the graph');
  reduceMotion=false;const prior=camera();jump('P2R0D');await new Promise(r=>setTimeout(r,80));const midway=camera();
  check(count()===total&&JSON.stringify(midway)!==JSON.stringify(prior),'Context jump animates through the continuous graph');
  await new Promise(r=>setTimeout(r,420));const final=camera(),p=full.get('P2R0D');
  check(Math.abs(p.x*final.k+final.x-size.width*.5)<.01,'Animated camera reaches the selected node');reduceMotion=true;
  click($('whole'));size={width:375,height:400};w.dispatchEvent(new w.Event('resize'));await new Promise(r=>setTimeout(r,25));
  check(count()===total&&M.bounds(full).width*camera().k<=375,'Mobile opens with the complete graph');
  click($('topics'));check(doc.body.classList.contains('detail-open'),'Mobile can open the reading guide');
  click($('detail').querySelector('[data-approaches]'));
  click($('detail').querySelector('.approach-list [data-jump="P2R0G"]'));check(count()===total,'Mobile selection retains the complete map');
  click($('close-detail'));check(!doc.body.classList.contains('detail-open'),'Mobile can return to the map');
  check(!$('minimap-wrap').hidden,'Mobile retains the overview locator');
  const deep='P2R0G19B_MECH_N0_1_OPEN_PROBLEM';w.location.hash='#node='+deep;await new Promise(r=>setTimeout(r,25));
  check(count()===total&&has(deep),'Deep links reveal context without hiding other branches');
  check($('detail').querySelector('h2').textContent===M.title(tree.byId.get(deep)),'Deep link opens the matching record');
  saveSVG('graph-mobile-context.svg');
  search('<img src=x onerror=alert(1)>');check(!$('results').querySelector('img'),'Search input cannot inject markup');key($('search'),'Escape');check($('results').hidden,'Escape dismisses search');
  check(errors.length===0,'No DOM runtime errors');
  fs.writeFileSync(path.join(root,'_build/viewer_dom_validation.json'),JSON.stringify({checks,engine:'jsdom 27.0.1',browser_rendered:false,embedded_html_executed:true,selection_keeps_full_graph:true,selection_preserves_every_node_position:true,released_path_keeps_full_graph:true,released_path_preserves_node_positions:true,development_route_reaches_problem:true,development_milestones:release.reading_order.length,development_reader_traversed:true,architecture_replacement_explained:true,detached_associations_removed:true,floating_topic_labels_removed:true,problem_first_reading_flow:true,six_approach_index:true,attempt_and_return_navigation:true,primary_links_open_work_directories:true,technical_details_closed_by_default:true,raw_research_notes_hidden_by_default:true,selected_outline_pulse:true,reduced_motion_respected:true,context_zoom_animated:true,clickable_minimap:true,back_forward_restores_camera:true,keyboard_search:true,mobile_navigation:true,exact_final_build_links:true,source_links_switch:true,runtime_errors:errors.length},null,2)+'\n');
  console.log(`${checks} DOM interaction checks passed; this does not verify browser rendering.`);w.close();
})().catch(e=>{console.error(e);w.close();process.exitCode=1;});
