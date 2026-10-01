'use client';

import { useEffect, useRef, useState } from 'react';
import { MessageCircle, Send, X } from 'lucide-react';
import { sendChat, type ChatTurn } from '../../api/backend';
import { LANGUAGE_OPTIONS, type LangCode, type Panchayat } from '../../types';

const SUGGESTIONS = [
  'How much rain is expected here?',
  'What should I do on my crop?',
  'How accurate is the model?',
  'Why do some panchayats have no value?',
];

const PROJECT_FACTS = `Answer only from these facts and the selected panchayat; if unknown, say so. Keep it to 2-4 sentences.
Rainfall refined from IMD 0.25 deg (28 km) to 0.05 deg (5 km) by a residual U-Net (117,329 parameters), deployed as an E+F ensemble, using SRTM elevation and ERA5-Land. Scored against CHIRPS, a reference not ground truth. 2022 test MAE 8.43 mm/day vs 9.60 baseline (-12.2%). Train 2018-20, validation 2021, test 2022. All 87,735 panchayats are mapped (direct grid, area-weighted, or nearest cell within 20 km). 3,389 sit on masked coastal cells and get no value. Nine deterministic rules decide the crop advisory (heavy rain at 64.5 mm/day or more); an LLM only rephrases it. Extreme Western Ghats rain is under-estimated.`;

export function ChatWidget({ selected }: { selected: Panchayat | null }) {
  const [open, setOpen] = useState(false);
  const [lang, setLang] = useState<LangCode>('en-IN');
  const [busy, setBusy] = useState(false);
  const [input, setInput] = useState('');
  const [turns, setTurns] = useState<ChatTurn[]>([]);
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => { endRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [turns, busy, open]);

  const ask = async (text: string) => {
    const q = text.trim();
    if (!q || busy) return;
    const history = turns;
    setTurns(t => [...t, { role: 'user', content: q }]);
    setInput('');
    setBusy(true);
    const ctx = [
      PROJECT_FACTS,
      selected
        ? `Selected panchayat: ${selected.panchayat_name}, ${selected.block_name}, ${selected.district}, ${selected.state}. Date ${selected.date}. Rainfall ${selected.rainfall_mm} mm/day, temperature ${selected.temperature_c ?? 'n/a'} C, humidity ${selected.humidity_pct ?? 'n/a'} %, elevation ${selected.elevation_m ?? 'n/a'} m.`
        : 'No panchayat is selected.',
      ...history.slice(-6).map(h => `${h.role === 'user' ? 'User' : 'Assistant'}: ${h.content}`),
    ].join('\n');
    try {
      const r = await sendChat(q, ctx, lang);
      setTurns(t => [...t, { role: 'assistant', content: r.answer }]);
    } catch {
      setTurns(t => [...t, { role: 'assistant', content: 'The assistant could not answer. It needs the backend running with a Groq key set.' }]);
    } finally {
      setBusy(false);
    }
  };

  return (
    <>
      {open && (
        <div
          className="fixed bottom-32 right-5 z-[1000] flex flex-col w-[22rem] max-w-[calc(100vw-2.5rem)] h-[30rem] max-h-[70vh]"
          style={{ background: 'var(--raised)', border: '1px solid var(--hairline)', borderRadius: '8px', boxShadow: '0 12px 40px rgba(0,0,0,0.45)' }}
        >
          <div className="flex items-center justify-between px-4 py-3" style={{ borderBottom: '1px solid var(--hairline)' }}>
            <div>
              <p className="text-sm font-bold" style={{ color: 'var(--text)' }}>Ask about rainfall</p>
              <p className="text-[0.65rem]" style={{ color: 'var(--muted)' }}>
                {selected ? `${selected.panchayat_name}, ${selected.district}` : 'No panchayat selected'}
              </p>
            </div>
            <div className="flex items-center gap-2">
              <select
                value={lang}
                onChange={e => setLang(e.target.value as LangCode)}
                aria-label="Reply language"
                className="text-[0.65rem] px-1 py-1"
                style={{ background: 'var(--panel)', color: 'var(--text-2)', border: '1px solid var(--hairline)', borderRadius: '4px' }}
              >
                {LANGUAGE_OPTIONS.map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
              </select>
              <button onClick={() => setOpen(false)} aria-label="Close chat" style={{ color: 'var(--muted)' }}><X className="w-4 h-4" /></button>
            </div>
          </div>

          <div className="flex-1 overflow-y-auto px-4 py-3 space-y-3">
            {turns.length === 0 && (
              <div>
                <p className="text-xs mb-3" style={{ color: 'var(--text-2)' }}>
                  Ask about the selected panchayat&apos;s rainfall and advisory, or how the model works.
                </p>
                <div className="flex flex-wrap gap-2">
                  {SUGGESTIONS.map(s => (
                    <button key={s} onClick={() => ask(s)} className="text-[0.7rem] px-2 py-1 text-left"
                      style={{ background: 'var(--panel)', color: 'var(--text-2)', border: '1px solid var(--hairline)', borderRadius: '999px' }}>
                      {s}
                    </button>
                  ))}
                </div>
              </div>
            )}
            {turns.map((t, i) => (
              <div key={i} className={t.role === 'user' ? 'flex justify-end' : 'flex'}>
                <p className="text-xs leading-relaxed px-3 py-2 max-w-[85%]"
                  style={{
                    background: t.role === 'user' ? 'var(--panel)' : 'transparent',
                    color: 'var(--text)',
                    border: '1px solid var(--hairline)',
                    borderRadius: '8px',
                  }}>
                  {t.content}
                </p>
              </div>
            ))}
            {busy && <p className="text-xs" style={{ color: 'var(--muted)' }}>Thinking…</p>}
            <div ref={endRef} />
          </div>

          <form onSubmit={e => { e.preventDefault(); ask(input); }} className="flex gap-2 p-3" style={{ borderTop: '1px solid var(--hairline)' }}>
            <input
              value={input}
              onChange={e => setInput(e.target.value)}
              placeholder="Type a question"
              maxLength={500}
              className="flex-1 text-xs px-3 py-2 outline-none"
              style={{ background: 'var(--panel)', color: 'var(--text)', border: '1px solid var(--hairline)', borderRadius: '4px' }}
            />
            <button type="submit" disabled={busy || !input.trim()} aria-label="Send"
              className="px-3 disabled:opacity-40" style={{ background: 'var(--panel)', color: 'var(--text)', border: '1px solid var(--hairline)', borderRadius: '4px' }}>
              <Send className="w-4 h-4" />
            </button>
          </form>
        </div>
      )}

      <button
        onClick={() => setOpen(o => !o)}
        aria-label={open ? 'Close chat' : 'Open chat'}
        className="fixed bottom-[4.5rem] right-5 z-[1000] h-11 px-4 flex items-center gap-2 text-xs font-semibold"
        style={{ background: 'var(--raised)', color: 'var(--text)', border: '1px solid var(--hairline-strong)', borderRadius: '999px', boxShadow: '0 6px 24px rgba(0,0,0,0.4)' }}
      >
        {open ? <X className="w-4 h-4" /> : <MessageCircle className="w-4 h-4" />}
        {open ? 'Close' : 'Ask AI'}
      </button>
    </>
  );
}
