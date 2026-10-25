// Regression checks for the actual browser controller; these do not test ASR.
const test = require('node:test');
const assert = require('node:assert/strict');
const vm = require('node:vm');
const fs = require('node:fs');
const path = require('node:path');
const controller = fs.readFileSync(path.join(__dirname,'../app/client.js'),'utf8');
const settle = async () => { for(let i=0;i<5;i++) await new Promise(setImmediate); };

async function harness() {
  const elements = new Map(), contexts = [], sockets = [];
  const element = {querySelector(selector) {
    if(!elements.has(selector)) elements.set(selector,{textContent:'',disabled:false});
    return elements.get(selector);
  }};
  class Socket {
    static OPEN=1;
    constructor() { this.readyState=1; this.sent=[]; this.bufferedAmount=0; sockets.push(this); setImmediate(()=>this.onopen?.()); }
    send(x) {this.sent.push(JSON.parse(x));}
  }
  class Context {
    constructor() { this.sampleRate=48000; this.state='running'; contexts.push(this); this.audioWorklet={addModule:async()=>{}}; }
    async resume() {}
    createGain() {return {gain:{value:0},connect(){},disconnect(){}};}
    createMediaStreamSource() {return {connect(){},disconnect(){}};}
    close() {this.state='closed'; return this.delayClose ? new Promise(r=>{this.resolveClose=r;}) : Promise.resolve();}
  }
  const box=vm.createContext({element,WebSocket:Socket,AudioContext:Context,
    AudioWorkletNode:class {constructor(){this.port={postMessage(){}};} connect(){} disconnect(){}},
    navigator:{mediaDevices:{getUserMedia:async()=>({getTracks:()=>[{stop(){}}]})}},
    location:{protocol:'http:',host:'127.0.0.1:7860'},
    URL:{createObjectURL:()=> 'blob:test',revokeObjectURL(){}},Blob,
    setTimeout:(...args)=>setTimeout(...args).unref(),
    requestAnimationFrame:cb=>setImmediate(cb),performance,
    window:{addEventListener(){}}});
  vm.runInContext(controller,box);
  await settle();
  const send=async data=>{await sockets[0].onmessage({data:JSON.stringify(data)}); await settle();};
  async function begin(id) {
    const started=elements.get('#dk-start').onclick(); await settle();
    await send({type:'started',session:id}); await started; return contexts.at(-1);
  }
  return {box,elements,contexts,sockets,send,begin};
}

test('old completed status cannot close the next capture before its new ID arrives', async()=>{
  const h=await harness();
  await h.begin('old');
  await h.send({session:'old',done:true,status:'Sesi berhenti',history:'previous session'});
  const starting=h.elements.get('#dk-start').onclick(); await settle();
  const next=h.contexts.at(-1);
  await h.send({session:'old',done:true,status:'Sesi berhenti',history:'stale text'});
  assert.equal(next.state,'running');
  assert.equal(h.elements.get('#dk-history').textContent,'');
  await h.send({type:'started',session:'new'}); await starting;
  assert.equal(vm.runInContext('active',h.box),true);
  await h.send({session:'new',done:true,status:'Sesi berhenti',history:'new text'});
});

test('an old asynchronous close cannot erase a new AudioContext', async()=>{
  const h=await harness();
  const old=await h.begin('old'); old.delayClose=true;
  const closing=vm.runInContext('cleanupAudio()',h.box);
  const next=vm.runInContext('context = new AudioContext()',h.box);
  old.resolveClose(); await closing;
  assert.equal(vm.runInContext('context',h.box),next);
  await vm.runInContext('cleanupAudio()',h.box);
});
