'use strict';
const $ = id => document.getElementById(id);
let csrf = '', user = null, authAction = 'login', source = 'upload', selectedFile = null;
let previewURL = null, stream = null, busy = false, cycling = false, previewBusy = false;
let latest = null, history = [], cameraEpoch = 0, runEpoch = 0;
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
function engineOptions(){
  const engine=$('engine').value;
  if(engine!=='original')$('mode').value='mediapipe';
  $('mode').disabled=engine!=='original';
  $('decoder').disabled=engine==='auto-avsr';
  $('duration').max=engine==='auto-avsr'?'16':'10';
  $('model-guidance').textContent=engine==='auto-avsr'?'Use one clear English sentence, up to 16 seconds. Processing may take longer on CPU. Names and unfamiliar phrases can be wrong.':engine==='lipnet-reference'?'GRID phrases follow: command, color, position, letter, number, ending. Example: set white with p two soon. Other speech can produce plausible but incorrect words.':'The original model is retained for comparison and failed the documented reference tests.';
}
$('engine').onchange=engineOptions;engineOptions();
function notice(message='') {$('notice').textContent=message; $('notice').hidden=!message;}
async function api(path, options={}) {
  const headers = new Headers(options.headers || {});
  if (options.method && options.method !== 'GET') headers.set('X-CSRF-Token', csrf);
  const response = await fetch(path, {...options, headers, credentials:'same-origin'});
  const data = await response.json();
  if (!response.ok) {
    if(response.status===401){stopCamera(); user=null; await sessionInfo();}
    throw new Error(data.message || ({busy:'The service is busy. Please try again shortly.',invalid_video:'This video could not be processed. Check framing, duration and file format.',unauthorized:'Please sign in.', 'Request Entity Too Large':'This upload is too large. Use a video under 30 MB.'}[data.error]) || data.error || 'Request failed');
  }
  return data;
}
async function sessionInfo() {
  const data = await api('/api/auth/session'); csrf=data.csrf; user=data.user;
  $('username').textContent=user?user.username:'Not signed in'; $('logout').hidden=!user;
  if(user){if($('auth-dialog').open)$('auth-dialog').close(); await loadHistory();}
  else if(!$('auth-dialog').open)$('auth-dialog').showModal();
  refreshButtons();
}
$('auth-dialog').addEventListener('cancel', e=>e.preventDefault());
$('auth-switch').onclick=()=>{authAction=authAction==='login'?'register':'login';$('auth-title').textContent=authAction==='login'?'Welcome back.':'Make it your space.';$('auth-submit').textContent=authAction==='login'?'Sign in':'Create account';$('auth-switch').textContent=authAction==='login'?'New here? Create an account':'Already registered? Sign in';$('auth-password').autocomplete=authAction==='login'?'current-password':'new-password';$('auth-error').textContent='';};
$('auth-form').onsubmit=async e=>{e.preventDefault();$('auth-submit').disabled=true;$('auth-error').textContent='';try{const data=await api('/api/auth/'+authAction,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({username:$('auth-username').value,password:$('auth-password').value})});csrf=data.csrf;$('auth-password').value='';await sessionInfo();}catch(error){$('auth-error').textContent=error.message;}finally{$('auth-submit').disabled=false;}};
$('logout').onclick=async()=>{stopCamera();try{await api('/api/auth/logout',{method:'POST'});history=[];latest=null;$('result-content').hidden=true;$('result-empty').hidden=false;await sessionInfo();}catch(e){notice(e.message);}};
function setView(view){document.querySelectorAll('.view').forEach(el=>el.hidden=el.id!==view);document.querySelectorAll('.nav').forEach(el=>el.classList.toggle('active',el.dataset.view===view));$('page-label').textContent={studio:'Recognition studio',history:'Prediction history',guide:'Getting started'}[view];if(view!=='studio')stopCamera();if(view==='history')loadHistory().catch(e=>notice(e.message));}
document.querySelectorAll('[data-view]').forEach(el=>el.onclick=()=>setView(el.dataset.view));
function refreshButtons(){ $('recognize').disabled=busy||!user||(source==='upload'?!selectedFile:!stream);$('recognize').textContent=busy?'Processing…':source==='upload'?'Recognize video ↗':'Recognize capture ↗';$('stop-live').hidden=!cycling;$('remove-file').disabled=busy;$('start-camera').disabled=busy||!!stream;$('stop-camera').disabled=!stream;$('upload-tab').disabled=busy;$('camera-tab').disabled=busy;}
function switchSource(next){if(busy)return;source=next;$('upload-pane').hidden=next!=='upload';$('camera-pane').hidden=next!=='camera';for(const id of ['upload','camera']){$(id+'-tab').classList.toggle('selected',next===id);$(id+'-tab').setAttribute('aria-selected',String(next===id));}if(next==='upload')stopCamera();notice();refreshButtons();}
$('upload-tab').onclick=()=>switchSource('upload');$('camera-tab').onclick=()=>switchSource('camera');
function selectFile(file){if(busy)return;notice();if(!file)return;if(file.size>30*1024*1024){notice('Choose a video smaller than 30 MB.');return;}if(!/\.(mp4|mpg|mpeg|avi)$/i.test(file.name)){notice('Choose an MP4, MPG, MPEG or AVI video.');return;}selectedFile=file;if(previewURL)URL.revokeObjectURL(previewURL);previewURL=URL.createObjectURL(file);$('uploaded-video').src=previewURL;$('file-name').textContent=file.name+' · '+(file.size/1024/1024).toFixed(1)+' MB';$('file-preview').hidden=false;$('dropzone').hidden=true;refreshButtons();}
$('video-file').onchange=e=>selectFile(e.target.files[0]);$('remove-file').onclick=()=>{selectedFile=null;if(previewURL)URL.revokeObjectURL(previewURL);previewURL=null;$('uploaded-video').removeAttribute('src');$('video-file').value='';$('file-preview').hidden=true;$('dropzone').hidden=false;refreshButtons();};
$('uploaded-video').onerror=()=>notice('This browser cannot preview this format. You can still submit the video for processing.');
for(const event of ['dragenter','dragover'])$('dropzone').addEventListener(event,e=>{e.preventDefault();$('dropzone').classList.add('drag');});
for(const event of ['dragleave','drop'])$('dropzone').addEventListener(event,e=>{e.preventDefault();$('dropzone').classList.remove('drag');if(event==='drop')selectFile(e.dataTransfer.files[0]);});
function setBusy(value){busy=value;refreshButtons();document.querySelector('.result-panel').classList.toggle('processing',value);if(value){$('result-content').hidden=true;$('result-empty').hidden=false;$('result-empty').querySelector('h3').textContent='Reading the movement…';$('result-empty').querySelector('p').textContent='Processing visual frames. No audio is used.';}}
function renderResult(result){
  latest=result;$('result-empty').hidden=true;$('result-content').hidden=false;
  $('result-status').textContent=result.usable_text?'Candidate · review required':'Unable to recognize';
  $('result-text').textContent=result.usable_text||'No reliable speech recognized.';
  $('result-message').textContent=result.message||'Review this candidate carefully.';
  let advice=$('capture-advice');
  if(!advice){advice=document.createElement('p');advice.id='capture-advice';advice.className='small';$('result-message').after(advice);}
  const tips=[...(result.input_quality?.warnings||[]),...(result.capture_timing?.warnings||[])];
  advice.textContent=[...new Set(tips)].join(' ');advice.hidden=!tips.length;
  $('raw-result').textContent=JSON.stringify({text:result.text,status:result.status,engine:result.engine,domain:result.domain,decoder:result.decoder,model_score:result.model_score,alternatives:result.alternatives,input_quality:result.input_quality,capture_timing:result.capture_timing,warnings:result.warnings},null,2);
}
function download(value,name){const url=URL.createObjectURL(new Blob([JSON.stringify(value,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
$('download-result').onclick=()=>latest&&download(latest,'visionspeak-result.json');
async function processInput(){if(busy)return;notice();setBusy(true);const engine=$('engine').value;try{let result;if(source==='upload'){const form=new FormData();form.append('video',selectedFile);form.append('engine',engine);form.append('mode',$('mode').value);form.append('decoder',$('decoder').value);form.append('start',$('start-time').value||'0');if($('duration').value)form.append('duration',$('duration').value);result=await api('/api/v1/recognize',{method:'POST',body:form});}else{const frames=await captureFrames();frames.engine=engine;result=await api('/api/v1/recognize-frames',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(frames)});}renderResult(result);await loadHistory();}catch(error){notice(error.message);cycling=false;if(latest)renderResult(latest);else{$('result-empty').querySelector('h3').textContent='Ready when you are.';$('result-empty').querySelector('p').textContent='Check the message above and try again.';}}finally{setBusy(false);}}
$('recognize').onclick=async()=>{const epoch=++runEpoch;cycling=source==='camera'&&$('live-mode').checked;refreshButtons();do{await processInput();if(cycling)await sleep(4000);}while(epoch===runEpoch&&cycling&&stream&&user);};
$('stop-live').onclick=()=>{runEpoch++;cycling=false;refreshButtons();notice('Live recognition stopped. Any current request will finish.');};
const frameCanvas=document.createElement('canvas');frameCanvas.width=640;frameCanvas.height=480;
function snapshot(){const video=$('camera-video');if(!stream||video.readyState<2)throw new Error('The camera is not ready.');const ratio=Math.min(640/video.videoWidth,480/video.videoHeight);frameCanvas.width=Math.round(video.videoWidth*ratio);frameCanvas.height=Math.round(video.videoHeight*ratio);frameCanvas.getContext('2d').drawImage(video,0,0,frameCanvas.width,frameCanvas.height);return frameCanvas.toDataURL('image/jpeg',.7).split(',')[1];}
async function captureFrames(){const epoch=cameraEpoch;const frames=[],timestamps_ms=[];const started=performance.now();for(let i=0;i<75;i++){await sleep(Math.max(0,started+i*40-performance.now()));if(epoch!==cameraEpoch||!stream||document.hidden)throw new Error('Capture stopped. Keep the camera on and this tab visible.');if(performance.now()-started>9500)throw new Error('Capture was too slow. Close other busy tabs and try again.');frames.push(snapshot());timestamps_ms.push(performance.now()-started);$('lip-status').textContent='Capturing '+(i+1)+' / 75';}return {frames,timestamps_ms};}
$('start-camera').onclick=async()=>{notice();try{if(!navigator.mediaDevices?.getUserMedia)throw new Error('Camera access requires HTTPS or localhost and a supported browser.');stream=await navigator.mediaDevices.getUserMedia({video:{width:{ideal:640},height:{ideal:480},frameRate:{ideal:25}},audio:false});$('camera-video').srcObject=stream;await $('camera-video').play();$('camera-placeholder').hidden=true;$('lip-status').textContent='Camera ready';cameraEpoch++;refreshButtons();}catch(error){stopCamera();notice(error.name==='NotAllowedError'?'Camera permission was denied. Allow camera access in your browser or upload a video.':error.message);}};
function stopCamera(){runEpoch++;cycling=false;cameraEpoch++;if(stream)stream.getTracks().forEach(track=>track.stop());stream=null;$('camera-video').srcObject=null;$('camera-placeholder').hidden=false;$('lip-status').textContent='Camera off';const canvas=$('lip-overlay');canvas.getContext('2d').clearRect(0,0,canvas.width,canvas.height);refreshButtons();}
$('stop-camera').onclick=stopCamera;window.addEventListener('pagehide',stopCamera);document.addEventListener('visibilitychange',()=>{if(document.hidden)stopCamera();});
setInterval(async()=>{if(!stream||busy||previewBusy||!user)return;previewBusy=true;const epoch=cameraEpoch;try{const data=await api('/api/v1/lips',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({image:snapshot()})});if(epoch!==cameraEpoch)return;$('lip-status').textContent=data.message;const canvas=$('lip-overlay');canvas.width=data.width||640;canvas.height=data.height||480;const ctx=canvas.getContext('2d');ctx.clearRect(0,0,canvas.width,canvas.height);if(data.box){const [x1,y1,x2,y2]=data.box;ctx.strokeStyle='#8cf4c8';ctx.lineWidth=3;ctx.strokeRect(x1,y1,x2-x1,y2-y1);}}catch(error){if(epoch===cameraEpoch)$('lip-status').textContent='Lip preview temporarily unavailable';}finally{previewBusy=false;}},900);
async function loadHistory(){if(!user)return;const data=await api('/api/v1/history');history=data.items;$('history-count').textContent=history.length;renderHistory();}
function renderHistory(){const filter=$('history-search').value.toLowerCase();const rows=history.filter(item=>((item.result.text||'')+' '+(item.result.source_label||'')).toLowerCase().includes(filter));$('history-items').replaceChildren();$('history-empty').hidden=rows.length>0;for(const item of rows){const row=document.createElement('article');row.className='history-row';const content=document.createElement('div');const title=document.createElement('strong');title.textContent=item.result.usable_text||'No reliable speech recognized';const meta=document.createElement('p');meta.textContent=(item.result.source_label||'Video')+' · '+new Date(item.created).toLocaleString();const state=document.createElement('p');state.textContent='Confidence not calibrated · '+(item.result.usable_text?'Review required':'Unreliable output');content.append(title,meta,state);const actions=document.createElement('div');actions.className='actions';const view=document.createElement('button');view.className='secondary';view.textContent='View result';view.onclick=()=>{renderResult(item.result);setView('studio');};const remove=document.createElement('button');remove.className='text-button danger';remove.textContent='Delete';remove.onclick=async()=>{if(!confirm('Delete this saved result?'))return;try{await api('/api/v1/history/'+item.id,{method:'DELETE'});await loadHistory();}catch(error){notice(error.message);}};actions.append(view,remove);row.append(content,actions);$('history-items').append(row);}}
$('history-search').oninput=renderHistory;$('export-history').onclick=()=>download(history,'visionspeak-history.json');$('clear-history').onclick=async()=>{if(!history.length||!confirm('Delete all your saved results? This cannot be undone.'))return;try{await api('/api/v1/history',{method:'DELETE'});await loadHistory();}catch(error){notice(error.message);}};
async function health(){try{const data=await api('/health/ready');$('health-label').textContent=data.status==='ready'?'Service ready':'Service starting';$('health-dot').classList.remove('off');}catch{$('health-label').textContent='Service unavailable';$('health-dot').classList.add('off');}}
sessionInfo().catch(error=>notice(error.message));health();setInterval(health,30000);

$('dropzone').addEventListener('keydown',event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();$('video-file').click();}});
