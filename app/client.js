// Gradio HTML js_on_load provides `element`. No Gradio Audio cumulative
// recording is used: this AudioWorklet emits incremental mono float32 PCM.
const el = (id) => element.querySelector('#dk-' + id);
let socket, context, media, node, source, silentGain;
let active = false, starting = false, stopping = false, session = null, lastAck = -1;
let resetPending = false, readyResolve = null, readyReject = null;
let awaitingNewSession = false;
const localSend = (value) => {
  if (socket?.readyState === WebSocket.OPEN) socket.send(JSON.stringify(value));
};
const status = (value) => { el('status').textContent = value; };
const controls = (running) => {
  el('start').disabled = running; el('replay').disabled = running;
  el('long').disabled = running; el('stop').disabled = !running;
};
async function cleanupAudio() {
  active = false;
  // Detach synchronously: an older async close must never null a new session's
  // AudioContext after its close() promise resolves.
  const oldContext=context, oldMedia=media, oldNode=node, oldSource=source, oldGain=silentGain;
  context = media = node = source = silentGain = null;
  if (oldMedia) oldMedia.getTracks().forEach(t => t.stop());
  if (oldSource) { try { oldSource.stop?.(); oldSource.disconnect(); } catch (_) {} }
  oldNode?.disconnect(); oldGain?.disconnect();
  if (oldContext && oldContext.state !== 'closed') await oldContext.close();
}
const workletCode = `
class ChunkCapture extends AudioWorkletProcessor {
  constructor() {
    super(); this.limit = Math.round(sampleRate / 4); this.buffer = new Float32Array(this.limit); this.used = 0; this.running = true;
    this.port.onmessage = e => { if (e.data === 'stop') { this.running = false; this.flush(); this.port.postMessage({stopped:true}); } };
  }
  flush() { if (this.used) { let packet=this.buffer.slice(0,this.used); this.port.postMessage(packet, [packet.buffer]); this.used=0; } }
  process(inputs) {
    if (!this.running) return true;
    const channels=inputs[0]; if (!channels?.length) return true;
    for (let i=0;i<channels[0].length;i++) {
      let value=0; for(let c=0;c<channels.length;c++) value+=channels[c][i]/channels.length;
      this.buffer[this.used++]=value;
      if(this.used===this.limit) this.flush();
    }
    return true;
  }
}
registerProcessor('dk-chunk-capture',ChunkCapture);`;
async function connect() {
  if (socket?.readyState === WebSocket.OPEN) return;
  socket = new WebSocket(`${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}/step2/audio`);
  await new Promise((resolve,reject) => {socket.onopen=resolve; socket.onerror=reject;});
  socket.onmessage = async event => {
    const data = JSON.parse(event.data);
    if(data.type === 'started') {session=data.session; awaitingNewSession=false; lastAck=-1; readyResolve?.(); readyResolve=readyReject=null; return;}
    // A stopped session can still have an already-published status in transit.
    // Ignore it during new capture setup, before the new ID is acknowledged.
    if(awaitingNewSession && data.session) return;
    if(data.session && session && data.session !== session) return;
    if(data.type === 'reset') {
      session=null; lastAck=-1; resetPending=false; stopping=false; controls(false);
      el('history').textContent=''; el('partial').textContent=''; el('timing').textContent='Belum ada audio.';
      el('session').textContent='';
    }
    if(data.error) { readyReject?.(new Error(data.error)); readyResolve=readyReject=null; status('Error: ' + data.error); await cleanupAudio(); controls(false); }
    else if(data.status) status(data.status);
    if(data.history !== undefined) el('history').textContent=data.history;
    if(data.partial !== undefined) el('partial').textContent=data.partial;
    if(data.audio_seconds !== undefined) {
      const proc=data.processing_seconds === null ? '—' : data.processing_seconds.toFixed(2)+' s';
      el('timing').textContent=`Audio ${data.audio_seconds.toFixed(2)} s · proses terakhir ${proc} · audio belum diproses ${data.lag_seconds.toFixed(2)} s`;
      el('session').textContent='Sesi: '+data.session;
    }
    if(data.update_id > lastAck && data.update_id > 0) {
      lastAck=data.update_id;
      requestAnimationFrame(() => requestAnimationFrame(() => localSend({type:'render_ack', session:data.session, update_id:data.update_id, browser_ms:performance.now()})));
    }
    if(data.done) { await cleanupAudio(); controls(false); stopping=false; }
  };
  socket.onclose = () => { cleanupAudio(); controls(false); status('Koneksi lokal terputus. Muat ulang halaman.'); };
}
async function start(replaySeconds=null) {
  if(active || starting || stopping || resetPending) return;
  starting=true;
  awaitingNewSession=true;
  el('reset').disabled=true;
  el('history').textContent=''; el('partial').textContent='';
  el('timing').textContent='Belum ada audio.'; el('session').textContent='';
  controls(true); status('Menyiapkan audio…');
  try {
    await connect();
    // Ask microphone permission before creating a server session.
    if(replaySeconds === null) media=await navigator.mediaDevices.getUserMedia({audio:{channelCount:1,echoCancellation:false,noiseSuppression:false,autoGainControl:false}});
    context=new AudioContext(); await context.resume();
    let replay;
    if(replaySeconds !== null) {
      const response=await fetch('/step2/fixture');
      if(!response.ok) throw new Error('Sampel replay belum tersedia. Jalankan download fixture.');
      const clip=await context.decodeAudioData(await response.arrayBuffer());
      const duration=replaySeconds || clip.duration;
      replay=context.createBuffer(clip.numberOfChannels, Math.ceil(duration*context.sampleRate), context.sampleRate);
      const unit=clip.length + context.sampleRate; // documented 1-second pause between repeats
      for(let c=0;c<clip.numberOfChannels;c++) {
        const dst=replay.getChannelData(c), input=clip.getChannelData(c);
        for(let p=0;p<dst.length;p+=unit) dst.set(input.subarray(0,Math.min(input.length,dst.length-p)),p);
      }
    }
    const blob=URL.createObjectURL(new Blob([workletCode],{type:'text/javascript'}));
    try {await context.audioWorklet.addModule(blob);} finally {URL.revokeObjectURL(blob);}
    node=new AudioWorkletNode(context,'dk-chunk-capture');
    silentGain=context.createGain(); silentGain.gain.value=0;
    node.connect(silentGain); silentGain.connect(context.destination);
    const ready=new Promise((resolve,reject) => {
      readyResolve=resolve;
      readyReject=reject;
      setTimeout(() => { if(readyResolve===resolve) {readyResolve=readyReject=null; reject(new Error('Model/sesi tidak siap. Periksa pesan error.'));} },10000);
    });
    localSend({type:'start',sample_rate:context.sampleRate,source:replay ? 'browser_paced_public_fixture' : 'browser_microphone'});
    await ready;
    active=true;
    node.port.onmessage = async event => {
      if(event.data.stopped) {
        localSend({type:'stop'}); await cleanupAudio(); return;
      }
      if(!active && !stopping) return;
      // Bounded browser transport; fail explicitly instead of hiding lost audio.
      if(socket.bufferedAmount > 256000) {
        localSend({type:'capture_error',message:'Audio transport backlog exceeded 256 kB'});
        status('Error: pengiriman audio tertinggal.'); await cleanupAudio(); controls(false); return;
      }
      socket.send(event.data.buffer);
    };
    if(replay) {
      source=context.createBufferSource(); source.buffer=replay; source.onended=() => stop();
      source.connect(node); source.start(); status('Replay berjalan sesuai waktu audio');
    } else {
      source=context.createMediaStreamSource(media); source.connect(node); status('Mikrofon aktif · menunggu cukup audio');
    }
  } catch(error) {
    awaitingNewSession=false;
    localSend({type:'capture_error',message:error.message});
    await cleanupAudio(); controls(false); status('Error: '+error.message+' Periksa izin mikrofon dan model lokal.');
  } finally {
    starting=false;
    el('reset').disabled=false;
  }
}
function stop() {
  if(!active || stopping) return;
  stopping=true; el('stop').disabled=true; status('Menyelesaikan audio terakhir…');
  node.port.postMessage('stop'); // final partial PCM packet arrives before stop marker
}
el('start').onclick=() => start(null);
el('stop').onclick=stop;
el('replay').onclick=() => start(0);
el('long').onclick=() => start(125);
el('reset').onclick=async () => {
  readyReject?.(new Error('Sesi direset')); readyResolve=readyReject=null;
  resetPending=true; await cleanupAudio(); controls(true);
  await connect(); localSend({type:'reset'});
};
connect().then(()=>status('Siap · izinkan mikrofon ketika diminta')).catch(()=>status('Aplikasi lokal tidak dapat dihubungi.'));
window.addEventListener('beforeunload',()=>{media?.getTracks().forEach(t=>t.stop()); socket?.close();});
