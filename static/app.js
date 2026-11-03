'use strict';
const $ = id => document.getElementById(id);
let csrf = '', user = null, authAction = 'login', source = 'upload', selectedFile = null;
let previewURL = null, stream = null, busy = false, cycling = false, previewBusy = false, activeRecorder = null;
let latest = null, history = [], cameraEpoch = 0, runEpoch = 0;
const researchOptions = [...$('engine').options].filter(option => ['original','lipnet-reference'].includes(option.value));
researchOptions.forEach(option => option.remove());
$('research-models').onchange=()=>{
  if(stream)stopCamera();
  if($('research-models').checked){researchOptions.forEach(option=>$('engine').append(option));}
  else{const wasResearch=researchOptions.some(option=>option.selected);researchOptions.forEach(option=>option.remove());if(wasResearch)$('engine').value=$('engine').querySelector('[value="auto-avsr"]')?'auto-avsr':'audio-video';speechEngine=$('engine').value;}
  engineOptions();
};
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
function engineOptions(){
  const engine=$('engine').value;
  if(engine!=='original')$('mode').value='mediapipe';
  $('mode').disabled=engine!=='original';
  $('decoder').disabled=['auto-avsr','audio-video','lip-expression'].includes(engine);
  $('engine').disabled=busy;
  $('research-settings').hidden=engine==='lip-expression';
  $('mode').parentElement.hidden=engine!=='original';
  $('decoder').parentElement.hidden=!['original','lipnet-reference'].includes(engine);
  $('duration').max=['audio-video','lip-expression'].includes(engine)?'300':engine==='auto-avsr'?'16':'10';
  const expression=engine==='lip-expression';
  if($('recognition-options-note'))$('recognition-options-note').textContent=expression?'Start and duration apply to uploads. Webcam records until you press Stop recording. Framing and speech decoder settings are not used.':'Framing, timing and decoder options apply to uploaded video. Webcam uses automatic detection and the standard decoder.';
  if(!latest&&!busy){
    $('result-empty').querySelector('h3').textContent=expression?'See what your lips are doing.':'Your words and lip expressions will appear here.';
    $('result-empty').querySelector('p').textContent=expression?'Try a smile or a pout. We’ll show the lip movements we notice.':'Choose a short clip when you’re ready. We’ll take it from there.';
  }
  refreshButtons();
  $('model-guidance').textContent=engine==='lip-expression'?'Recognize smiling, pouting, pursed, closed, slightly open and open lips in a timeline. Record until you press Stop recording. Clips can be up to five minutes. These are visible shapes, not emotions.':engine==='audio-video'?'Transcribe speech locally and capture lip shapes. Record an English sentence with your microphone, then press Stop recording. Clips can be up to five minutes. Spoken words come from the audio; lip shapes are shown separately.':engine==='auto-avsr'?'Use one clear English sentence, up to 16 seconds. The first request loads the model. Later requests reuse it, but CPU processing can still take longer than the clip. Names and unfamiliar phrases can be wrong.':engine==='lipnet-reference'?'GRID phrases follow: command, color, position, letter, number, ending. Example: set white with p two soon. Other speech can produce plausible but incorrect words.':'The original model is retained for comparison and failed the documented reference tests.';
}
let speechEngine=$('engine').value;
$('feature').onchange=()=>{if(stream)stopCamera();if($('feature').value==='expression'){speechEngine=$('engine').value;$('engine').value='lip-expression';}else{$('engine').value=speechEngine;}$('engine-label').hidden=$('feature').value==='expression';engineOptions();};
$('engine').onchange=()=>{if(stream)stopCamera();engineOptions();};engineOptions();
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
$('auth-switch').onclick=()=>{authAction=authAction==='login'?'register':'login';$('auth-title').textContent=authAction==='login'?'Hello, welcome in.':'Let’s get you started.';$('auth-submit').textContent=authAction==='login'?'Sign in':'Create account';$('auth-switch').textContent=authAction==='login'?'New here? Create an account':'Already registered? Sign in';$('auth-password').autocomplete=authAction==='login'?'current-password':'new-password';$('auth-error').textContent='';};
$('auth-form').onsubmit=async e=>{e.preventDefault();$('auth-submit').disabled=true;$('auth-error').textContent='';try{const data=await api('/api/auth/'+authAction,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({username:$('auth-username').value,password:$('auth-password').value})});csrf=data.csrf;$('auth-password').value='';await sessionInfo();}catch(error){$('auth-error').textContent=error.message;}finally{$('auth-submit').disabled=false;}};
$('logout').onclick=async()=>{stopCamera();try{await api('/api/auth/logout',{method:'POST'});history=[];latest=null;$('result-content').hidden=true;$('result-empty').hidden=false;await sessionInfo();}catch(e){notice(e.message);}};
function setView(view){document.querySelectorAll('.view').forEach(el=>el.hidden=el.id!==view);document.querySelectorAll('.nav').forEach(el=>el.classList.toggle('active',el.dataset.view===view));$('page-label').textContent={studio:'Try it out',history:'Your past results',guide:'Getting started'}[view];if(view!=='studio')stopCamera();if(view==='history')loadHistory().catch(e=>notice(e.message));}
document.querySelectorAll('[data-view]').forEach(el=>el.onclick=()=>setView(el.dataset.view));
function refreshButtons(){ $('recognize').disabled=busy||cycling||!user||(source==='upload'?!selectedFile:!stream);$('recognize').textContent=busy?'Processing…':$('engine').value==='lip-expression'?'Explore these lip gestures ↗':source==='upload'?'Read words & expressions ↗':'Record and read ↗';$('stop-live').hidden=!cycling;$('stop-recording').hidden=!activeRecorder||activeRecorder.state==='inactive';$('remove-file').disabled=busy;$('start-camera').disabled=busy||!!stream;$('stop-camera').disabled=!stream;$('upload-tab').disabled=busy;$('camera-tab').disabled=busy;}
function switchSource(next){if(busy)return;source=next;$('upload-pane').hidden=next!=='upload';$('camera-pane').hidden=next!=='camera';for(const id of ['upload','camera']){$(id+'-tab').classList.toggle('selected',next===id);$(id+'-tab').setAttribute('aria-selected',String(next===id));}if(next==='upload')stopCamera();notice();refreshButtons();}
$('upload-tab').onclick=()=>switchSource('upload');$('camera-tab').onclick=()=>switchSource('camera');
function selectFile(file){if(busy)return;notice();if(!file)return;if(file.size>30*1024*1024){notice('Choose a video smaller than 30 MB.');return;}if(!/\.(mp4|webm|mpg|mpeg|avi)$/i.test(file.name)){notice('Choose an MP4, WebM, MPG, MPEG or AVI video.');return;}selectedFile=file;if(previewURL)URL.revokeObjectURL(previewURL);previewURL=URL.createObjectURL(file);$('uploaded-video').src=previewURL;$('file-name').textContent=file.name+' · '+(file.size/1024/1024).toFixed(1)+' MB';$('file-preview').hidden=false;$('dropzone').hidden=true;refreshButtons();}
$('video-file').onchange=e=>selectFile(e.target.files[0]);$('remove-file').onclick=()=>{selectedFile=null;if(previewURL)URL.revokeObjectURL(previewURL);previewURL=null;$('uploaded-video').removeAttribute('src');$('video-file').value='';$('file-preview').hidden=true;$('dropzone').hidden=false;refreshButtons();};
$('uploaded-video').onerror=()=>notice('This browser cannot preview this format. You can still submit the video for processing.');
for(const event of ['dragenter','dragover'])$('dropzone').addEventListener(event,e=>{e.preventDefault();$('dropzone').classList.add('drag');});
for(const event of ['dragleave','drop'])$('dropzone').addEventListener(event,e=>{e.preventDefault();$('dropzone').classList.remove('drag');if(event==='drop')selectFile(e.dataTransfer.files[0]);});
function setBusy(value){busy=value;$('research-models').disabled=value;$('engine').disabled=value;$('feature').disabled=value;refreshButtons();document.querySelector('.result-panel').classList.toggle('processing',value);if(value){$('result-content').hidden=true;$('result-empty').hidden=false;$('result-empty').querySelector('h3').textContent='Taking a closer look…';$('result-empty').querySelector('p').textContent=$('engine').value==='audio-video'?'Listening to your clip and looking at lip movements…':$('engine').value==='auto-avsr'?'Reading your English sentence. The first request loads the model; later requests reuse it. Please wait…':'Looking at the lip movements in your clip…';}}
function renderResult(result){
  $('result-source').textContent=result.engine==='audio-video'?'Source: audio transcription — words came from sound.':result.engine==='lip-expression'?'Source: visible lip shapes — no words transcribed.':'Source: visual-only lip-reading — audio was not used.';
  latest=result;$('result-empty').hidden=true;$('result-content').hidden=false;
  $('download-result').disabled=false;
  const expression=result.engine==='lip-expression';
  $('speech-confidence').hidden=expression;
  $('result-status').textContent=expression?(result.dominant_expression?'Lip shapes measured':'Face unavailable'):result.usable_text?'A possible reading · please check':'Unable to recognize';
  $('result-text').textContent=resultTitle(result);
  $('result-message').textContent=result.message||'Review this candidate carefully.';
  let advice=$('capture-advice');
  if(!advice){advice=document.createElement('p');advice.id='capture-advice';advice.className='small';$('result-message').after(advice);}
  const tips=[...(result.input_quality?.warnings||[]),...(result.capture_timing?.warnings||[])];
  advice.textContent=[...new Set(tips)].join(' ');advice.hidden=!tips.length;
  renderSpeechDetails(result);
  $('raw-result').textContent=JSON.stringify({words:result.words,lip_expressions:result.lip_expressions,audio_used:result.audio_used,language:result.language,text:result.text,status:result.status,engine:result.engine,domain:result.domain,decoder:result.decoder,model_score:result.model_score,alternatives:result.alternatives,input_quality:result.input_quality,capture_timing:result.capture_timing,warnings:result.warnings},null,2);
}
function download(value,name){const url=URL.createObjectURL(new Blob([JSON.stringify(value,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
$('download-result').onclick=()=>latest&&download(latest,'visionspeak-result.json');
async function processInput(){
  if(busy)return;
  latest=null;
  $('result-text').textContent='';
  $('download-result').disabled=true;
  notice();setBusy(true);
  const engine=$('engine').value;
  try{
    const form=new FormData();
    if(source==='camera'){
      const recording=await captureAudioVideo(engine==='audio-video');
      form.append('video',recording.blob,recording.name);
      // Recorder elapsed time includes startup/finalization overhead. Let the
      // decoder read the real clip instead of selecting past its last frame.
    }else{
      form.append('video',selectedFile);
      form.append('mode',$('mode').value);
      form.append('decoder',$('decoder').value);
      form.append('start',$('start-time').value||'0');
      if($('duration').value)form.append('duration',$('duration').value);
    }
    form.append('engine',engine);
    const result=await api('/api/v1/recognize',{method:'POST',body:form});
    renderResult(result);
    $('download-result').disabled=false;
    try{await loadHistory();}catch{notice('Your new result is ready. History could not refresh right now.');}
  }catch(error){
    notice(error.message);cycling=false;
    $('result-content').hidden=true;
    $('result-empty').hidden=false;
    $('result-empty').querySelector('h3').textContent='This recording couldn’t be read.';
    $('result-empty').querySelector('p').textContent=error.message;
  }finally{setBusy(false);}
}
$('recognize').onclick=async()=>{const epoch=++runEpoch;cycling=source==='camera'&&$('live-mode').checked;refreshButtons();do{await processInput();if(cycling)await sleep(500);}while(epoch===runEpoch&&cycling&&stream&&user);};
$('stop-live').onclick=()=>{runEpoch++;cycling=false;refreshButtons();notice('Live recognition stopped. Any current request will finish.');};
const frameCanvas=document.createElement('canvas');frameCanvas.width=640;frameCanvas.height=480;
function snapshot(){const video=$('camera-video');if(!stream||video.readyState<2)throw new Error('The camera is not ready.');const ratio=Math.min(640/video.videoWidth,480/video.videoHeight);frameCanvas.width=Math.round(video.videoWidth*ratio);frameCanvas.height=Math.round(video.videoHeight*ratio);frameCanvas.getContext('2d').drawImage(video,0,0,frameCanvas.width,frameCanvas.height);return frameCanvas.toDataURL('image/jpeg',.7).split(',')[1];}
$('start-camera').onclick=async()=>{notice();try{if(!navigator.mediaDevices?.getUserMedia)throw new Error('Camera access requires HTTPS or localhost and a supported browser.');stream=await navigator.mediaDevices.getUserMedia({video:{width:{ideal:640},height:{ideal:480},frameRate:{ideal:25}},audio:$('engine').value==='audio-video'});$('camera-video').srcObject=stream;await $('camera-video').play();$('camera-placeholder').hidden=true;$('lip-status').textContent='Camera ready';cameraEpoch++;refreshButtons();}catch(error){stopCamera();notice(error.name==='NotAllowedError'?'Camera or microphone permission was denied. Allow the requested permissions or upload a video.':error.message);}};
function stopCamera(){runEpoch++;cycling=false;cameraEpoch++;if(activeRecorder&&activeRecorder.state!=='inactive')activeRecorder.stop();if(stream)stream.getTracks().forEach(track=>track.stop());stream=null;$('camera-video').srcObject=null;$('camera-placeholder').hidden=false;$('lip-status').textContent='Camera off';const canvas=$('lip-overlay');canvas.getContext('2d').clearRect(0,0,canvas.width,canvas.height);refreshButtons();}
$('stop-camera').onclick=stopCamera;window.addEventListener('pagehide',stopCamera);document.addEventListener('visibilitychange',()=>{if(document.hidden)stopCamera();});
$('stop-recording').onclick=()=>{cycling=false;if(activeRecorder&&activeRecorder.state!=='inactive'){activeRecorder.stop();$('lip-status').textContent='Finishing recording…';}refreshButtons();};
setInterval(async()=>{if(!stream||busy||cycling||previewBusy||!user)return;previewBusy=true;const epoch=cameraEpoch;try{const data=await api('/api/v1/lips',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({image:snapshot()})});if(epoch!==cameraEpoch)return;$('lip-status').textContent=data.message;const canvas=$('lip-overlay');canvas.width=data.width||640;canvas.height=data.height||480;const ctx=canvas.getContext('2d');ctx.clearRect(0,0,canvas.width,canvas.height);if(data.box){const [x1,y1,x2,y2]=data.box;ctx.strokeStyle='#8cf4c8';ctx.lineWidth=3;ctx.strokeRect(x1,y1,x2-x1,y2-y1);}}catch(error){if(epoch===cameraEpoch)$('lip-status').textContent='Lip preview temporarily unavailable';}finally{previewBusy=false;}},900);
async function loadHistory(){if(!user)return;const data=await api('/api/v1/history');history=data.items;$('history-count').textContent=history.length;renderHistory();}
function renderHistory(){const filter=$('history-search').value.toLowerCase();const rows=history.filter(item=>(((item.result.text||'')+' '+(item.result.dominant_expression||''))+' '+(item.result.source_label||'')).toLowerCase().includes(filter));$('history-items').replaceChildren();$('history-empty').hidden=rows.length>0;for(const item of rows){const row=document.createElement('article');row.className='history-row';const content=document.createElement('div');const title=document.createElement('strong');title.textContent=resultTitle(item.result);const meta=document.createElement('p');meta.textContent=(item.result.source_label||'Video')+' · '+new Date(item.created).toLocaleString();const state=document.createElement('p');state.textContent=item.result.engine==='lip-expression'?'Lip expression recognition · Visible gesture estimate':'Confidence not calibrated · '+(item.result.usable_text?'Review required':'Unreliable output');content.append(title,meta,state);const actions=document.createElement('div');actions.className='actions';const view=document.createElement('button');view.className='secondary';view.textContent='View result';view.onclick=()=>{renderResult(item.result);setView('studio');};const remove=document.createElement('button');remove.className='text-button danger';remove.textContent='Delete';remove.onclick=async()=>{if(!confirm('Delete this saved result?'))return;try{await api('/api/v1/history/'+item.id,{method:'DELETE'});await loadHistory();}catch(error){notice(error.message);}};actions.append(view,remove);row.append(content,actions);$('history-items').append(row);}}
$('history-search').oninput=renderHistory;$('export-history').onclick=()=>download(history,'visionspeak-history.json');$('clear-history').onclick=async()=>{if(!history.length||!confirm('Delete all your saved results? This cannot be undone.'))return;try{await api('/api/v1/history',{method:'DELETE'});await loadHistory();}catch(error){notice(error.message);}};
async function health(){try{const data=await api('/health/ready');$('health-label').textContent=data.status==='ready'?'Service ready':'Service starting';$('health-dot').classList.remove('off');}catch{$('health-label').textContent='Service unavailable';$('health-dot').classList.add('off');}}
sessionInfo().catch(error=>notice(error.message));health();setInterval(health,30000);

$('dropzone').addEventListener('keydown',event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();$('video-file').click();}});

function resultTitle(result){return result.engine==='lip-expression'?(result.dominant_expression?'Mostly '+result.dominant_expression:'No lip expressions detected.'):result.usable_text||(result.engine==='original'?'The original checkpoint did not produce usable text.':'We couldn’t make out the words this time.');}
function renderSpeechDetails(result){
  $('spoken-details').hidden=!(result.words||[]).length;
  $('expression-heading').textContent=result.dominant_expression?'Lip expressions · mostly '+result.dominant_expression:'Lip expressions';
  $('speech-details').hidden=!result.lip_expressions;
  $('word-timeline').textContent=(result.words||[]).map(w=>`${w.word} (${w.start.toFixed(1)}–${w.end.toFixed(1)}s)`).join(' · ')||'No spoken words detected.';
  $('expression-note').textContent=result.lip_expressions?.description||'';
  if(result.engine==='lip-expression'){
    const shapes=result.lip_expressions;
    $('expression-note').textContent+=` Face detected in ${shapes.detected_samples} of ${shapes.total_samples} samples.`;
    if(shapes.gesture_samples===0&&shapes.detected_samples>0)$('expression-note').textContent+=' Gesture model output unavailable; only lip opening was measured.';
  }
  const summary=$('expression-summary');summary.replaceChildren();
  for(const [label,count] of Object.entries(result.lip_expressions?.summary||{})){
    const item=document.createElement('span');item.className='expression-sample';item.textContent=`${label}: ${count} samples`;summary.append(item);
  }
  const container=$('expression-timeline');container.replaceChildren();
  for(const sample of result.lip_expressions?.samples||[]){
    const item=document.createElement('span');item.className='expression-sample';
    item.textContent=`${sample.time.toFixed(1)}s · ${sample.label}`;
    if(sample.opening_label&&sample.opening_label!==sample.label)item.title=sample.opening_label;
    container.append(item);
  }
}
async function captureAudioVideo(withAudio=true){
  if(!window.MediaRecorder)throw new Error('This browser cannot record video. Upload a video instead.');
  if(withAudio&&!stream?.getAudioTracks().some(track=>track.readyState==='live'))throw new Error('Enable the camera again and allow microphone access for audio mode.');
  const epoch=cameraEpoch;
  const mime=(withAudio?['video/webm;codecs=vp8,opus','video/webm','video/mp4']:['video/webm;codecs=vp8','video/webm','video/mp4']).find(type=>MediaRecorder.isTypeSupported(type));
  if(!mime)throw new Error('No supported recording format. Upload a video instead.');
  const recorder=new MediaRecorder(stream,{mimeType:mime,videoBitsPerSecond:1500000});
  const chunks=[];
  let recordedBytes=0;
  const completed=new Promise((resolve,reject)=>{
    recorder.ondataavailable=e=>{if(e.data.size){chunks.push(e.data);recordedBytes+=e.data.size;if(recordedBytes>28*1024*1024&&recorder.state!=='inactive'){notice('Recording reached the file-size limit. Processing the captured clip.');recorder.stop();}}};
    recorder.onerror=()=>reject(new Error('Recording failed. Try again or upload a video.'));
    recorder.onstop=resolve;
  });
  // Attach immediately so a device error during the capture cannot reject unhandled.
  completed.catch(()=>{});
  const started=performance.now();
  recorder.start(1000);
  const timer=setInterval(()=>{const seconds=Math.floor((performance.now()-started)/1000);$('lip-status').textContent=`Recording · ${Math.floor(seconds/60)}:${String(seconds%60).padStart(2,'0')} · press Stop when ready`;},500);
  activeRecorder=recorder;
  $('lip-status').textContent=`Recording ${withAudio?'video and audio':'video'}…`;
  $('result-empty').querySelector('h3').textContent='Recording your clip…';
  $('result-empty').querySelector('p').textContent='Finish your sentence, then press Stop recording. Keep this tab open.';
  refreshButtons();
  try{
    await completed;
    if(epoch!==cameraEpoch||document.hidden)throw new Error('Recording stopped. Keep the camera on and this tab visible.');
  }finally{
    try{
      if(recorder.state!=='inactive')recorder.stop();
      await completed;
    }finally{
      clearInterval(timer);
      if(activeRecorder===recorder)activeRecorder=null;
      refreshButtons();
    }
  }
  $('lip-status').textContent='Recording complete';
  $('result-empty').querySelector('h3').textContent='Reading your clip…';
  $('result-empty').querySelector('p').textContent='Your result will appear here when it’s ready.';
  const blob=new Blob(chunks,{type:mime});
  if(!blob.size)throw new Error('The recording was empty. Please try again.');
  const duration=(performance.now()-started)/1000;
  const engine=$('engine').value;
  const limit=['audio-video','lip-expression'].includes(engine)?300:engine==='auto-avsr'?16:10;
  if(duration>limit)throw new Error(`This mode supports clips up to ${limit} seconds. For longer English recordings, choose With sound.`);
  return {blob,duration,name:mime.startsWith('video/mp4')?'camera.mp4':'camera.webm'};
}
