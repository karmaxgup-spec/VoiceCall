import { extension_settings, getContext } from '../../../extensions.js';
import { eventSource, event_types, saveSettingsDebounced } from '../../../../script.js';

const KEY = 'phone_call';
const PROMPT_KEY = 'phone_call_mode';
const CH_MS = 64; // one worklet frame at 16 kHz (1024 samples)
const SAMPLE_RATE = 16000;
const SCHEMA = 2; // bump when defaults change so old settings migrate
const STT_FALLBACK_PATHS = [
    '/scripts/extensions/third-party/Extension-Speech-Recognition/moonshine-local.js',
    '/scripts/extensions/speech-recognition/moonshine-local.js',
];

const DEFAULTS = {
    schema: SCHEMA,
    enabled: true,
    allowAiToCall: true,
    sttModulePath: STT_FALLBACK_PATHS[0],
    sttModel: 'UsefulSensors/moonshine-streaming-tiny',
    silenceMs: 600,
    bargeInMs: 250,
    vadThreshold: 0.02,
    ringSeconds: 25,
};

const settings = () => {
    const s = (extension_settings[KEY] ??= structuredClone(DEFAULTS));
    if (s.schema !== SCHEMA) {
        // Old Whisper/niagara values would break the Moonshine provider, so reset them.
        s.sttModulePath = DEFAULTS.sttModulePath;
        s.sttModel = DEFAULTS.sttModel;
        s.schema = SCHEMA;
    }
    for (const k of Object.keys(DEFAULTS)) if (!(k in s)) s[k] = DEFAULTS[k];
    delete s.useWebGPU;
    return s;
};
const ctx = () => getContext();
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const slashSafe = (t) => String(t).replace(/[|\r\n]+/g, ' ').trim();
const fmt = (sec) => `${Math.floor(sec / 60)}:${String(sec % 60).padStart(2, '0')}`;

// ---------- state ----------
const S = {
    state: 'idle', // idle | outgoing | incoming | connecting | connected
    muted: false,
    awaitingReply: false,
    streamSpokenLen: 0,
    gen: 0,
    speakQueue: [],
    speaking: false,
    timerStart: 0,
    timerIv: null,
    micStream: null,
    audioCtx: null,
    worklet: null,
    sttReady: false,
    sttLoading: null,
    incomingMsg: '',
    ringTimeout: null,
    ringCtx: null,
    ringIv: null,
};

// ---------- speech recognition (Moonshine provider from the Speech Recognition extension) ----------
let sttProvider = null;

async function getSttProvider() {
    const s = settings();
    if (!sttProvider) {
        const paths = [...new Set([s.sttModulePath, ...STT_FALLBACK_PATHS].filter(Boolean))];
        const errors = [];
        for (const path of paths) {
            try {
                const mod = await import(path);
                if (!mod.MoonshineLocalSttProvider) throw new Error('no MoonshineLocalSttProvider export');
                sttProvider = new mod.MoonshineLocalSttProvider();
                console.debug('[st-phone-call] speech provider loaded from', path);
                break;
            } catch (err) {
                errors.push(`${path}: ${err?.message ?? err}`);
            }
        }
        if (!sttProvider) throw new Error('Moonshine provider not found. Tried: ' + errors.join(' | '));
    }
    // Re-apply settings every call so model changes take effect without a reload.
    sttProvider.loadSettings({ language: '', model: s.sttModel });
    return sttProvider;
}

// Moonshine's processAudio() takes a Blob, so wrap the 16 kHz mono Float32 frames in a WAV.
function encodeWav(samples, sampleRate = SAMPLE_RATE) {
    const buf = new ArrayBuffer(44 + samples.length * 2);
    const v = new DataView(buf);
    const write = (o, str) => { for (let i = 0; i < str.length; i++) v.setUint8(o + i, str.charCodeAt(i)); };
    write(0, 'RIFF');
    v.setUint32(4, 36 + samples.length * 2, true);
    write(8, 'WAVE');
    write(12, 'fmt ');
    v.setUint32(16, 16, true);
    v.setUint16(20, 1, true); // PCM
    v.setUint16(22, 1, true); // mono
    v.setUint32(24, sampleRate, true);
    v.setUint32(28, sampleRate * 2, true);
    v.setUint16(32, 2, true);
    v.setUint16(34, 16, true);
    write(36, 'data');
    v.setUint32(40, samples.length * 2, true);
    let o = 44;
    for (let i = 0; i < samples.length; i++, o += 2) {
        const s = Math.max(-1, Math.min(1, samples[i]));
        v.setInt16(o, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    }
    return new Blob([buf], { type: 'audio/wav' });
}

async function transcribeSamples(samples) {
    const provider = await getSttProvider();
    const text = await provider.processAudio(encodeWav(samples));
    return String(text || '').trim();
}

// Loads the Moonshine pipeline by transcribing half a second of silence. Errors show up here, not mid-call.
function loadStt() {
    if (S.sttReady) return Promise.resolve(true);
    if (S.sttLoading) return S.sttLoading;
    S.sttLoading = (async () => {
        try {
            await transcribeSamples(new Float32Array(SAMPLE_RATE / 2));
            S.sttReady = true;
            return true;
        } catch (err) {
            console.error('[st-phone-call] speech model failed to load:', err);
            toast(`Speech model failed: ${err?.message ?? err}`, 'error');
            return false;
        } finally {
            S.sttLoading = null;
        }
    })();
    return S.sttLoading;
}

const HALLUCINATIONS = new Set(['thank you', 'thanks for watching', 'you', 'bye', 'the end', 'thanks', 'thank you very much']);
function isRealSpeech(t) {
    if (!t || t.length < 2) return false;
    if (/^[\[(].*[\])]$/.test(t)) return false; // [BLANK_AUDIO], (music)
    return !HALLUCINATIONS.has(t.toLowerCase().replace(/[.!?,]/g, '').trim());
}

// ---------- microphone + VAD (AudioWorklet) ----------
const WORKLET = `
class PhoneCap extends AudioWorkletProcessor {
  constructor() { super(); this.buf = []; this.n = 0; }
  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (ch) {
      this.buf.push(new Float32Array(ch)); this.n += ch.length;
      if (this.n >= 1024) {
        const out = new Float32Array(this.n); let o = 0;
        for (const b of this.buf) { out.set(b, o); o += b.length; }
        this.buf = []; this.n = 0;
        this.port.postMessage(out, [out.buffer]);
      }
    }
    return true;
  }
}
registerProcessor('phone-cap', PhoneCap);`;

async function startMic() {
    if (S.micStream) return;
    S.micStream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true, channelCount: 1 },
    });
    const ac = new AudioContext({ sampleRate: SAMPLE_RATE });
    S.audioCtx = ac;
    const url = URL.createObjectURL(new Blob([WORKLET], { type: 'application/javascript' }));
    await ac.audioWorklet.addModule(url);
    URL.revokeObjectURL(url);
    const node = new AudioWorkletNode(ac, 'phone-cap');
    const mute = ac.createGain();
    mute.gain.value = 0; // keeps the graph pulling without playing the mic back
    ac.createMediaStreamSource(S.micStream).connect(node);
    node.connect(mute).connect(ac.destination);
    node.port.onmessage = (e) => vad(e.data);
    S.worklet = node;
}

function stopMic() {
    S.micStream?.getTracks().forEach((t) => t.stop());
    S.audioCtx?.close();
    S.micStream = null;
    S.audioCtx = null;
    S.worklet = null;
    V.inSpeech = false;
    V.chunks = [];
    V.pre = [];
    V.preLen = 0;
    V.onset = 0;
    V.silence = 0;
    document.querySelector('#pc-overlay')?.classList.remove('pc-listening');
}

const V = { inSpeech: false, onset: 0, silence: 0, chunks: [], pre: [], preLen: 0, dur: 0 };

function vad(frame) {
    if (S.muted || S.state !== 'connected') return;
    const s = settings();
    let sum = 0;
    for (let i = 0; i < frame.length; i++) sum += frame[i] * frame[i];
    const loud = Math.sqrt(sum / frame.length) > s.vadThreshold;

    if (!V.inSpeech) {
        V.pre.push(frame);
        V.preLen += frame.length;
        while (V.preLen > SAMPLE_RATE * 0.3) { const f = V.pre.shift(); V.preLen -= f.length; }
        if (loud) {
            V.onset += CH_MS;
            // Need sustained speech to barge in on the AI; a short blip starts a normal turn.
            const need = aiBusy() ? s.bargeInMs : 150;
            if (V.onset >= need) {
                if (aiBusy()) stopSpeaking();
                V.inSpeech = true;
                V.chunks = [...V.pre];
                V.pre = [];
                V.preLen = 0;
                V.silence = 0;
                V.dur = 0;
                document.querySelector('#pc-overlay')?.classList.add('pc-listening');
            }
        } else V.onset = 0;
    } else {
        V.chunks.push(frame);
        V.dur += CH_MS;
        if (loud) V.silence = 0; else V.silence += CH_MS;
        if (V.silence >= s.silenceMs || V.dur > 20000) endUtterance();
    }
}

function endUtterance() {
    const chunks = V.chunks;
    V.inSpeech = false;
    V.chunks = [];
    V.onset = 0;
    V.silence = 0;
    document.querySelector('#pc-overlay')?.classList.remove('pc-listening');
    const total = chunks.reduce((a, c) => a + c.length, 0);
    if (total < SAMPLE_RATE * 0.4) return; // under 0.4s is noise
    const audio = new Float32Array(total);
    let o = 0;
    for (const c of chunks) { audio.set(c, o); o += c.length; }
    transcribeSamples(audio)
        .catch((err) => { console.error('[st-phone-call] transcribe failed:', err); return ''; })
        .then((text) => {
            if (!isRealSpeech(text) || S.state !== 'connected') return;
            addCaption('you', text);
            sendTurn(text);
        });
}

// ---------- speaking (SillyTavern TTS) ----------
function cleanForTts(text) {
    return String(text)
        .replace(/```[\s\S]*?```/g, '')
        .replace(/\*[^*\n]*\*/g, '') // *stage directions* are not spoken
        .replace(/[*_#`~\[\]]/g, '')
        .replace(/\s+/g, ' ')
        .trim();
}

function speak(text) {
    const t = cleanForTts(text);
    if (!t) return;
    S.speakQueue.push({ t, gen: S.gen });
    pump();
}

async function pump() {
    if (S.speaking) return;
    S.speaking = true;
    document.querySelector('#pc-overlay')?.classList.add('pc-speaking');
    while (S.speakQueue.length) {
        const item = S.speakQueue.shift();
        if (item.gen !== S.gen) continue;
        addCaption('char', item.t);
        await speakViaSt(item.t, item.gen);
    }
    S.speaking = false;
    document.querySelector('#pc-overlay')?.classList.remove('pc-speaking');
}

async function speakViaSt(text, gen) {
    await ctx().executeSlashCommandsWithOptions(`/speak ${slashSafe(text)}`);
    await waitForAudioIdle(gen);
}

// Waits for TTS audio to start and then finish. Gives up if it never starts.
function waitForAudioIdle(gen) {
    return new Promise((res) => {
        const t0 = performance.now();
        let seen = false;
        const iv = setInterval(() => {
            const playing = [...document.querySelectorAll('audio')].some((a) => !a.paused && !a.ended);
            if (playing) seen = true;
            const done = (seen && !playing) || (!seen && performance.now() - t0 > 3000) || performance.now() - t0 > 60000;
            if (done || gen !== S.gen) { clearInterval(iv); res(); }
        }, 80);
    });
}

function stopSpeaking() {
    S.gen++;
    S.speakQueue.length = 0;
    S.awaitingReply = false;
    document.querySelectorAll('audio').forEach((a) => { try { a.pause(); } catch { /* ignore */ } });
    const stop = document.querySelector('#mes_stop');
    if (stop && stop.offsetParent) stop.click(); // stop an in-flight generation
    document.querySelector('#pc-overlay')?.classList.remove('pc-speaking');
}

const aiBusy = () => S.awaitingReply || S.speaking || S.speakQueue.length > 0;

// ---------- chat turns ----------
function beginReply() {
    S.awaitingReply = true;
    S.streamSpokenLen = 0;
}

async function sendTurn(text) {
    const c = ctx();
    beginReply();
    await c.executeSlashCommandsWithOptions(`/send ${slashSafe(text)}`);
    await c.executeSlashCommandsWithOptions('/trigger');
}

// Speak each finished sentence while the model is still streaming.
eventSource.on(event_types.STREAM_TOKEN_RECEIVED, (text) => {
    if (!S.awaitingReply || typeof text !== 'string') return;
    const pending = text.slice(S.streamSpokenLen);
    const matches = [...pending.matchAll(/[.!?…]+["”')*]*(\s|$)/g)];
    if (!matches.length) return;
    const last = matches[matches.length - 1];
    const end = last.index + last[0].length;
    const chunk = pending.slice(0, end);
    if (chunk.trim().length < 30) return; // avoid choppy one-word utterances
    S.streamSpokenLen += end;
    speak(chunk);
});

// Speak whatever the stream did not already cover.
eventSource.on(event_types.CHARACTER_MESSAGE_RENDERED, () => {
    if (!S.awaitingReply) return;
    const c = ctx();
    const full = c.chat[c.chat.length - 1]?.mes || '';
    const rest = full.slice(S.streamSpokenLen);
    S.awaitingReply = false;
    S.streamSpokenLen = 0;
    if (rest.trim()) speak(rest);
});

eventSource.on(event_types.GENERATION_STOPPED, () => {
    S.awaitingReply = false;
});

// ---------- call state ----------
function setExt(on) {
    const c = ctx();
    if (!c.setExtensionPrompt) return;
    c.setExtensionPrompt(
        PROMPT_KEY,
        on
            ? 'You are on a live phone call with the user. Reply in one to three short, natural spoken sentences. No action asterisks, no markdown, no lists. Respond to what they just said.'
            : '',
        c.extension_prompt_types?.IN_CHAT ?? 1,
        1,
    );
}

async function startOutgoing() {
    if (S.state !== 'idle') return;
    S.state = 'outgoing';
    buildUi();
    showCall('outgoing');
    ring(true);
    loadStt(); // load in the background; don't block the call on it
    await sleep(2000);
    ring(false);
    if (S.state !== 'outgoing') return;
    await connect();
    if (!S.sttReady) setStatus('Loading speech model…');
    const c = ctx();
    beginReply();
    await c.executeSlashCommandsWithOptions(`/sys ${slashSafe('[Phone call connected. You picked up. Greet the user briefly, in character.]')}`);
    await c.executeSlashCommandsWithOptions('/trigger');
}

// Called by the tool when the model rings the user.
function receiveCall(message) {
    if (S.state !== 'idle') return false;
    S.state = 'incoming';
    S.incomingMsg = message;
    buildUi();
    showCall('incoming');
    ring(true);
    S.ringTimeout = setTimeout(() => { if (S.state === 'incoming') missedCall(); }, settings().ringSeconds * 1000);
    loadStt(); // warm the model while it rings
    return true;
}

async function answerIncoming() {
    if (S.state !== 'incoming') return;
    clearTimeout(S.ringTimeout);
    ring(false);
    S.state = 'connecting';
    setStatus('Connecting…');
    const ready = await loadStt();
    if (!ready) { hideUi(); S.state = 'idle'; return; }
    await connect();
    speak(S.incomingMsg);
}

function declineIncoming() {
    clearTimeout(S.ringTimeout);
    ring(false);
    hideUi();
    S.state = 'idle';
    ctx().executeSlashCommandsWithOptions(`/sys ${slashSafe('[The user declined the call.]')}`);
}

function missedCall() {
    clearTimeout(S.ringTimeout);
    ring(false);
    hideUi();
    S.state = 'idle';
    ctx().executeSlashCommandsWithOptions(`/sys ${slashSafe('[Missed call from ' + ctx().name2 + '.]')}`);
}

async function connect() {
    S.state = 'connected';
    S.muted = false;
    setUiMode('active');
    startTimer();
    try {
        await startMic();
    } catch (err) {
        toast('Microphone unavailable: ' + err.message, 'error');
        endCall();
        return;
    }
    setExt(true);
}

function endCall() {
    const wasConnected = S.state === 'connected';
    const dur = S.timerStart ? fmt(Math.floor((Date.now() - S.timerStart) / 1000)) : '';
    stopSpeaking();
    ring(false);
    stopMic();
    clearInterval(S.timerIv);
    S.timerIv = null;
    S.timerStart = 0;
    S.state = 'idle';
    setExt(false);
    hideUi();
    if (wasConnected) {
        ctx().executeSlashCommandsWithOptions(`/sys ${slashSafe('[Call ended after ' + dur + '.]')}`);
    }
}

function startTimer() {
    S.timerStart = Date.now();
    S.timerIv = setInterval(() => {
        const sec = Math.floor((Date.now() - S.timerStart) / 1000);
        setText('.pc-timer', fmt(sec));
        setText('.pc-clock', new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }));
    }, 500);
}

// ---------- ringing ----------
function ring(on) {
    if (!on) {
        clearInterval(S.ringIv);
        S.ringIv = null;
        S.ringCtx?.close();
        S.ringCtx = null;
        return;
    }
    S.ringCtx = new AudioContext();
    const beep = () => {
        const o = S.ringCtx.createOscillator();
        const g = S.ringCtx.createGain();
        o.frequency.value = 440;
        g.gain.value = 0.08;
        o.connect(g).connect(S.ringCtx.destination);
        o.start();
        o.stop(S.ringCtx.currentTime + 1);
    };
    beep();
    S.ringIv = setInterval(beep, 3000);
}

// ---------- phone UI ----------
function buildUi() {
    if (document.getElementById('pc-overlay')) return;
    const d = document.createElement('div');
    d.id = 'pc-overlay';
    d.className = 'pc-hidden';
    d.innerHTML = `
      <div class="pc-phone">
        <div class="pc-topbar"><span class="pc-clock"></span><span class="pc-net">5G ▮</span></div>
        <div class="pc-call">
          <img class="pc-avatar" alt="">
          <div class="pc-name"></div>
          <div class="pc-state">Calling…</div>
          <div class="pc-timer"></div>
          <div class="pc-caption"></div>
        </div>
        <div class="pc-wave"><i></i><i></i><i></i><i></i><i></i></div>
        <div class="pc-row pc-incoming">
          <button class="pc-btn pc-decline" title="Decline">✕</button>
          <button class="pc-btn pc-answer" title="Answer">✆</button>
        </div>
        <div class="pc-row pc-active">
          <button class="pc-btn pc-mute" title="Mute">🎙</button>
          <button class="pc-btn pc-end" title="End call">✕</button>
        </div>
      </div>`;
    document.body.appendChild(d);
    d.querySelector('.pc-decline').onclick = () => declineIncoming();
    d.querySelector('.pc-answer').onclick = () => answerIncoming();
    d.querySelector('.pc-end').onclick = () => endCall();
    d.querySelector('.pc-mute').onclick = (e) => {
        S.muted = !S.muted;
        e.currentTarget.classList.toggle('pc-on', S.muted);
        if (S.muted) V.inSpeech = false;
    };
}

function showCall(mode) {
    buildUi();
    const c = ctx();
    const ch = c.characters?.[c.characterId];
    const img = document.querySelector('#pc-overlay .pc-avatar');
    if (ch?.avatar) img.src = `/characters/${encodeURIComponent(ch.avatar)}`;
    setText('.pc-name', c.name2 || 'Unknown');
    setText('.pc-caption', '');
    setStatus(mode === 'incoming' ? 'Incoming call' : 'Calling…');
    setUiMode(mode === 'incoming' ? 'incoming' : 'outgoing');
    document.getElementById('pc-overlay').classList.remove('pc-hidden');
}

function setUiMode(mode) {
    const o = document.getElementById('pc-overlay');
    if (!o) return;
    o.dataset.mode = mode;
    o.querySelector('.pc-incoming').style.display = mode === 'incoming' ? 'flex' : 'none';
    o.querySelector('.pc-active').style.display = mode === 'active' ? 'flex' : 'none';
    if (mode === 'active') setStatus('Connected');
}

function hideUi() {
    document.getElementById('pc-overlay')?.classList.add('pc-hidden');
    setText('.pc-timer', '');
}

function setStatus(t) { setText('.pc-state', t); }
function setText(sel, t) { const el = document.querySelector(`#pc-overlay ${sel}`); if (el) el.textContent = t; }

function addCaption(who, text) {
    const el = document.querySelector('#pc-overlay .pc-caption');
    if (!el) return;
    el.textContent = (who === 'you' ? 'You: ' : '') + text;
}

// Shows a toastr notification. type is 'info', 'success', 'warning', or 'error'.
function toast(msg, type = 'error') {
    if (typeof window.toastr !== 'undefined') window.toastr[type](msg);
    else console.log('[st-phone-call]', msg);
}

// ---------- tool: the model can ring the user ----------
function registerTool() {
    const c = ctx();
    if (!c.registerFunctionTool) {
        console.warn('[st-phone-call] registerFunctionTool not in context; AI calling disabled.');
        return;
    }
    c.registerFunctionTool({
        name: 'phone_call_user',
        displayName: 'Phone call the user',
        description:
            'Phone the user. Use sparingly, only when a call fits the story (urgent news, a late-night thought, missing them). ' +
            'The call rings on the user\'s screen. If they answer, `message` is spoken aloud as your opening line. ' +
            'Write `message` as spoken dialogue only, with no action text.',
        parameters: {
            $schema: 'http://json-schema.org/draft-04/schema#',
            type: 'object',
            properties: {
                message: { type: 'string', description: 'Your opening line, exactly as it will be spoken aloud.' },
            },
            required: ['message'],
        },
        action: async ({ message }) => {
            if (!settings().enabled || !settings().allowAiToCall) return 'The user has disabled calls.';
            const ok = receiveCall(String(message || '').slice(0, 600));
            return ok
                ? 'Call is ringing on the user\'s phone. Do not write any more dialogue for the call; wait for the user to answer.'
                : 'The user is already on another call.';
        },
        formatMessage: () => '📞 Calling user…',
        shouldRegister: () => settings().enabled && settings().allowAiToCall,
        stealth: false,
    });
}

// ---------- send-bar button + settings panel ----------
function addCallButton() {
    if (document.getElementById('pc_call_btn')) return;
    const host = document.querySelector('#leftSendForm') || document.querySelector('#send_form');
    if (!host) return;
    const b = document.createElement('div');
    b.id = 'pc_call_btn';
    b.className = 'fa-solid fa-phone interactable pc-call-btn';
    b.title = 'Call character';
    b.onclick = () => (S.state === 'idle' ? startOutgoing() : endCall());
    host.prepend(b);
}

function buildSettings() {
    const host = document.getElementById('extensions_settings2') || document.getElementById('extensions_settings');
    if (!host || document.getElementById('pc_settings')) return;
    const s = settings();
    const d = document.createElement('div');
    d.id = 'pc_settings';
    d.className = 'inline-drawer';
    d.innerHTML = `
      <div class="inline-drawer-toggle inline-drawer-header"><b>Phone Call</b>
        <div class="inline-drawer-icon fa-solid fa-circle-chevron-down down"></div></div>
      <div class="inline-drawer-content">
        <label class="checkbox_label"><input type="checkbox" data-k="enabled"> Enable phone calls</label>
        <label class="checkbox_label"><input type="checkbox" data-k="allowAiToCall"> Let the character call me</label>
        <label>Speech Recognition provider file (path to moonshine-local.js)<input class="text_pole" data-k="sttModulePath" type="text"></label>
        <label>Moonshine model<input class="text_pole" data-k="sttModel" type="text"></label>
        <label>Silence before reply (ms)<input class="text_pole" data-k="silenceMs" type="number"></label>
        <label>Barge-in speech (ms)<input class="text_pole" data-k="bargeInMs" type="number"></label>
        <label>Mic threshold (0–0.2)<input class="text_pole" data-k="vadThreshold" type="number" step="0.005"></label>
        <label>Ring duration (s)<input class="text_pole" data-k="ringSeconds" type="number"></label>
        <div class="flex-container">
          <input class="menu_button" type="button" id="pc_preload" value="Load speech model">
        </div>
      </div>`;
    host.appendChild(d);

    for (const el of d.querySelectorAll('[data-k]')) {
        const k = el.dataset.k;
        if (el.type === 'checkbox') el.checked = !!s[k];
        else el.value = s[k];
        el.addEventListener('change', () => {
            const v = el.type === 'checkbox' ? el.checked : el.type === 'number' ? Number(el.value) : el.value;
            s[k] = v;
            if (k === 'sttModel' || k === 'sttModulePath') {
                S.sttReady = false; // reload on next call
                sttProvider = null;
            }
            saveSettingsDebounced();
        });
    }
    d.querySelector('#pc_preload').onclick = async () => {
        toast('Loading speech model…', 'info');
        const ok = await loadStt();
        if (ok) toast('Speech model ready', 'success');
    };
}

// ---------- init ----------
jQuery(() => {
    settings();
    buildSettings();
    buildUi();
    addCallButton();
    registerTool();
    eventSource.on(event_types.CHAT_CHANGED, addCallButton);
    setTimeout(addCallButton, 1500); // send bar may render after load
});
