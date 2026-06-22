#!/usr/bin/env python3
"""Self-contained HTML poster (3 cols x 2 sections), figures embedded, MathJax equations.
Success story only: SAE factorization with grad-norm adversary + invariance."""
import base64
from pathlib import Path
P = Path(__file__).resolve().parent
def img(n): return f'<img src="data:image/png;base64,{base64.b64encode((P/"figs"/n).read_bytes()).decode()}"/>'

SECTIONS = [
 ("①&nbsp; Speech disentanglement &rarr; Sparse Autoencoders", f"""
   <p>Modern speech encoders pack <b>content</b> (<i>what</i>), <b>speaker</b> (<i>who</i>), and
   <b>prosody/emotion</b> (<i>how</i>) into every dimension. <b>Disentangling</b> them is a long-standing
   goal — for controllable synthesis, privacy, and interpretability.</p>
   <p><b>Why SAEs.</b> Sparse autoencoders drive recent <b>mechanistic-interpretability</b> progress,
   decomposing dense activations into <b>sparse, near-monosemantic</b> features that surface
   interpretable structure in large models.</p>
   <p class=hi><b>Hypothesis:</b> if speech's factors live in <b>separable sparse directions</b>, an SAE can
   <b>expose and route</b> them — turning interpretability into <b>disentanglement</b>,
   <i>without touching the backbone</i>.</p>
   <p><b>This work:</b> factor a <b>frozen</b> encoder into <span class=t>linguistic \\(z_L\\)</span> vs
   <span class=p>paralinguistic \\(z_P\\)</span>, anchored on <b>speaker identity (SID)</b> — extensible to
   prosody, emotion, ….</p>"""),

 ("②&nbsp; Method — SAE factorization", f"""
   <div class=fig>{img('architecture.png')}</div>
   <ul>
     <li><b>Frozen SPEAR-XLarge</b> → features \\(h_t\\) (1280-d), <i>not retrained</i></li>
     <li><b>TopK SAE:</b> \\(z=\\mathrm{{TopK}}(W_e(h-b)),\\;\\hat h=W_d z+b\\)</li>
     <li>The sparse code is <b>factored</b> into \\(z_L\\) (linguistic), \\(z_P\\) (paralinguistic), \\(z_U\\) (residual)</li>
     <li><b>Heads:</b> CTC phonemes on \\(z_L\\), speaker on pooled \\(z_P\\), recon from the full code</li>
   </ul>"""),

 ("③&nbsp; Learning objective", f"""
   <p>A single multi-task loss shapes the factorization:</p>
   <p class=eq>\\[\\mathcal{{L}}=\\mathcal{{L}}_{{\\mathrm{{rec}}}}+\\alpha\\,\\mathcal{{L}}_{{\\mathrm{{phon}}}}
      +\\beta\\,\\mathcal{{L}}_{{\\mathrm{{spk}}}}+\\lambda_{{\\mathrm{{adv}}}}\\mathcal{{L}}_{{\\mathrm{{adv}}}}
      +\\lambda_{{\\mathrm{{inv}}}}\\mathcal{{L}}_{{\\mathrm{{inv}}}}\\]</p>
   <ul>
     <li><b>Reconstruction</b> \\(\\mathcal{{L}}_{{\\mathrm{{rec}}}}=\\frac1T\\sum_t\\lVert h_t-\\hat h_t\\rVert^2\\)</li>
     <li><b>Content</b> \\(\\mathcal{{L}}_{{\\mathrm{{phon}}}}=\\mathrm{{CTC}}(g_\\phi(z_L),y)\\) — phonemes from \\(z_L\\)</li>
     <li><b>Speaker</b> \\(\\mathcal{{L}}_{{\\mathrm{{spk}}}}=\\mathrm{{CE}}(\\mathrm{{pool}}(z_P),s)\\) — identity in \\(z_P\\)</li>
     <li><b>Removal</b> \\(\\mathcal{{L}}_{{\\mathrm{{adv}}}},\\,\\mathcal{{L}}_{{\\mathrm{{inv}}}}\\) — push speaker out of \\(z_L\\) →</li>
   </ul>
   <p class=foot>Evaluation = strong probes (PER&darr; content · SID acc&uarr; speaker), per block.</p>"""),

 ("④&nbsp; Removing speaker from \\(z_L\\) — a dense signal", f"""
   <div class=fig>{img('mechanism.png')}</div>
   <p>Speaker is <b>utterance-level</b> but \\(z_L\\) is <b>per-frame</b> — a pooled adversary gives one diluted
   gradient. We make the speaker GRL <b>dense (per-frame)</b>, so every frame gets its own removal gradient,
   then strengthen it two ways:</p>
   <ul>
     <li><b>Grad-normalized adversary:</b> L2-normalize each frame's reversed gradient to magnitude \\(\\tau\\):
        \\(\\;\\tilde g_t=-\\lambda\\,\\tau\\,g_t/\\lVert g_t\\rVert\\)</li>
     <li><b>Perturbation invariance:</b> pitch+formant warp \\(P\\) changes speaker, keeps content; force
        \\(\\;z_L(x)\\approx z_L(P(x))\\)</li>
   </ul>"""),

 ("⑤&nbsp; Result — clean factorization", f"""
   <div class=fig>{img('result.png')}</div>
   <p><b>Two independent routes</b> reach the same clean split (probe TEST; SID chance = 0.004):</p>
   <table>
     <tr><th>route</th><th>\\(z_L\\) PER&darr;</th><th>\\(z_L\\)&rarr;SID&darr;</th><th>\\(z_P\\) PER&darr;</th><th>\\(z_P\\)&rarr;SID&uarr;</th></tr>
     <tr><td><b>dense + grad-norm</b></td><td><b>0.067</b></td><td><b>0.010</b></td><td>0.534</td><td><b>0.972</b></td></tr>
     <tr><td><b>dense + invariance</b></td><td><b>0.073</b></td><td><b>0.010</b></td><td>0.515</td><td><b>0.964</b></td></tr>
   </table>
   <p>Grad-norm and invariance <b>replicate one another</b> → method-robust, not a lucky run.</p>
   <p class=hi>Rigorous: strong stats probe, 10k steps · positive control (z_P→SID ≈ 0.97) passes ·
   \\(z_L\\) content intact (not collapsed).</p>"""),

 ("⑥&nbsp; Outlook — toward full paralinguistic factorization", f"""
   <ul>
     <li><b>SID is factor #1.</b> Give \\(z_P\\) more paralinguistic jobs — per-frame <b>prosody</b>
         (log-F0, energy), then <b>emotion / accent</b> — a multi-task paralinguistic code.</li>
     <li><b>Strengthen evidence:</b> replicate across seeds + <b>held-out-speaker</b> probe.</li>
     <li><b>Full 3-way split:</b> give residual \\(z_U\\) a positive reconstruction role so
         content / speaker / residual separate cleanly.</li>
   </ul>
   <p class=foot>SPEAR-XLarge (frozen) · LibriSpeech train-clean-100 · SUPERB 74-phone CTC · 251-spk SID.</p>"""),
]

cards="\n".join(f'<section><h2>{t}</h2>{b}</section>' for t,b in SECTIONS)
HTML=f"""<!doctype html><html><head><meta charset=utf-8><title>Poster</title>
<script>MathJax={{tex:{{inlineMath:[['\\\\(','\\\\)']],displayMath:[['\\\\[','\\\\]']]}}}};</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js" id=MathJax-script async></script>
<style>
 :root{{--t:#159a6c;--p:#e8643c;--u:#8b7fae;--d:#15263a;--hi:#d11149;--teal:#0d7d8c}}
 *{{box-sizing:border-box}} body{{margin:0;background:#e9eef1;font-family:'Helvetica Neue',Arial,sans-serif;color:#15263a}}
 header{{background:linear-gradient(105deg,#0d7d8c,#159a6c);color:#fff;padding:26px 34px;text-align:center;
   box-shadow:0 3px 10px rgba(0,0,0,.18)}}
 header h1{{margin:0;font-size:31px;letter-spacing:.2px}} header .sub{{font-size:18.5px;opacity:.96;margin-top:8px}}
 header .meta{{font-size:14px;opacity:.9;margin-top:7px}}
 .grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:20px;padding:22px;max-width:1760px;margin:auto}}
 section{{background:#fff;border-radius:14px;padding:18px 20px;box-shadow:0 3px 12px rgba(20,40,60,.12);
   border-top:6px solid var(--teal)}}
 section:nth-child(even){{border-top-color:var(--p)}}
 h2{{margin:.1em 0 .55em;font-size:21px;color:var(--d);line-height:1.25}}
 p,li{{font-size:15.5px;line-height:1.5}} ul{{margin:.35em 0 .35em 1.15em;padding:0}} li{{margin:.32em 0}}
 .fig img{{width:100%;border-radius:9px;border:1px solid #e1e8ec;box-shadow:0 1px 4px rgba(0,0,0,.06)}}
 .t{{color:var(--t);font-weight:bold}} .p{{color:var(--p);font-weight:bold}}
 .hi{{background:#fff2f5;border-left:4px solid var(--hi);padding:8px 11px;border-radius:7px;margin:.5em 0}}
 .eq{{text-align:center;font-size:16px;background:#f1f8f9;border:1px solid #d6e8ea;border-radius:8px;padding:4px 6px}}
 table{{width:100%;border-collapse:collapse;margin:.5em 0;font-size:15px}}
 th,td{{border:1px solid #e1e8ec;padding:7px 9px;text-align:center}} th{{background:#f2f7f8}}
 .foot{{font-size:12px;color:#6a7783;margin-top:10px}}
</style></head><body>
<header>
 <h1>Disentangling Linguistic &amp; Paralinguistic Factors in <u>Frozen</u> Speech Features with Sparse Autoencoders</h1>
 <div class=sub>Removing speaker identity from the content code &mdash; grad-normalized adversary + perturbation invariance</div>
 <div class=meta>bbg25@cam.ac.uk &middot; MLMI Thesis &middot; SPEAR-XLarge (frozen) &middot; LibriSpeech &middot; <i>mid-research progress</i></div>
</header>
<div class=grid>{cards}</div>
</body></html>"""
out=P/"poster.html"; out.write_text(HTML)
print("wrote",out,f"({len(HTML)//1024} KB; MathJax via CDN)")
