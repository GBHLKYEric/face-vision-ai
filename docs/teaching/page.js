(function(){
  var nav=document.querySelector('nav.top');
  var tabRow=nav.querySelector('.tabs');
  var tabs=[].slice.call(tabRow.querySelectorAll('a'));
  var target=function(a){return document.getElementById(a.getAttribute('href').slice(1));};
  var secs=tabs.map(target);
  var rows=tabs.map(function(a){return nav.querySelector('.parts[data-for="'+a.getAttribute('href').slice(1)+'"]');});
  var parts=rows.map(function(r){return [].slice.call(r.querySelectorAll('a')).map(function(a){return {a:a,el:target(a)};});});
  var cur=0,curPart=null,queued=false;

  function center(row,a){
    var left=a.offsetLeft-(row.clientWidth-a.offsetWidth)/2;
    row.scrollLeft=Math.max(0,left);
  }
  function measure(){
    document.documentElement.style.setProperty('--navh',Math.ceil(nav.getBoundingClientRect().height)+'px');
  }
  function spy(){
    queued=false;
    var line=nav.getBoundingClientRect().bottom+14;
    var i=0,k;
    for(k=0;k<secs.length;k++){if(secs[k].getBoundingClientRect().top<=line)i=k;}
    if(i!==cur){
      tabs[cur].removeAttribute('aria-current');rows[cur].hidden=true;
      tabs[i].setAttribute('aria-current','true');rows[i].hidden=false;
      cur=i;curPart=null;
      center(tabRow,tabs[i]);
    }
    var act=null;
    for(k=0;k<parts[i].length;k++){if(parts[i][k].el.getBoundingClientRect().top<=line)act=parts[i][k].a;}
    if(act!==curPart){
      if(curPart)curPart.removeAttribute('aria-current');
      if(act){act.setAttribute('aria-current','true');center(rows[i],act);}
      else rows[i].scrollLeft=0;
      curPart=act;
    }
  }
  function queue(){if(!queued){queued=true;requestAnimationFrame(spy);}}
  var eqs=[].slice.call(document.querySelectorAll('.eq'));
  function markEq(){
    eqs.forEach(function(e){e.classList.remove('of');});
    var over=eqs.map(function(e){return e.scrollWidth>e.clientWidth+1;});
    eqs.forEach(function(e,i){if(over[i])e.classList.add('of');});
  }

  document.addEventListener('click',function(ev){
    var a=ev.target.closest?ev.target.closest('a[href^="#"]'):null;
    if(!a)return;
    var id=a.getAttribute('href').slice(1),el=document.getElementById(id);
    if(!el)return;
    ev.preventDefault();
    el.scrollIntoView({block:'start'});
    try{history.replaceState(null,'','#'+id);}catch(e){}
    queue();
  });
  window.addEventListener('scroll',queue,{passive:true});
  window.addEventListener('resize',function(){measure();markEq();queue();});
  measure();markEq();spy();
  if(document.fonts&&document.fonts.ready)document.fonts.ready.then(function(){measure();markEq();queue();});
})();
