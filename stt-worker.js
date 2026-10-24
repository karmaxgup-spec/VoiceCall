import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3/+esm';

env.allowLocalModels = false; // fetch from the Hugging Face hub; the browser caches it after the first download

let asr = null;

self.onmessage = async (e) => {
    const m = e.data;

    if (m.type === 'load') {
        try {
            asr = await pipeline('automatic-speech-recognition', m.model, {
                device: m.device,
                dtype: m.dtype,
                progress_callback: (p) => self.postMessage({ type: 'progress', p: p.progress ?? 0 }),
            });
            // Warm-up: compiles kernels so the first real utterance is not slow.
            await asr(new Float32Array(8000));
            self.postMessage({ type: 'ready' });
        } catch (err) {
            self.postMessage({ type: 'error', error: String(err) });
        }
    }

    if (m.type === 'transcribe') {
        try {
            const out = await asr(m.audio, { num_beams: 1, return_timestamps: false });
            self.postMessage({ type: 'text', id: m.id, text: out.text || '' });
        } catch {
            self.postMessage({ type: 'text', id: m.id, text: '' });
        }
    }
}; 
