_STYLES = """
:root {
  color-scheme: light dark;
  --ink: #13231a;
  --ink-soft: #304137;
  --muted: #687267;
  --paper: #f3ecd9;
  --card: #f8f2e2;
  --line: #25382c;
  --line-soft: #b9b39f;
  --acid: #d7ff45;
  --accent: #d5522f;
  --accent-dark: #91351f;
  --positive: #285b3d;
  --negative: #b53d28;
  --header: rgba(243,236,217,.96);
  --assessment: #e6dfca;
  --assessment-ink: #304137;
  --annotation: #eee7d4;
  --control: #fffaf0;
  --control-line: #657064;
  --button-ink: #fffaf0;
  --grid-line: rgba(19,35,26,.055);
  --shadow: #13231a;
  --font-display: Georgia, "Times New Roman", Times, serif;
  --font-reading: Arial, Helvetica, ui-sans-serif, system-ui, sans-serif;
  --font-mono: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
  color: var(--ink);
  background: var(--paper);
  font-family: var(--font-reading);
}
* { box-sizing: border-box; }
html { min-width: 280px; }
body {
  margin: 0;
  background-color: var(--paper);
  background-image:
    linear-gradient(var(--grid-line) 1px, transparent 1px),
    linear-gradient(90deg, var(--grid-line) 1px, transparent 1px);
  background-size: 24px 24px;
  color: var(--ink);
  font-family: var(--font-reading);
  line-height: 1.55;
}
::selection { background: var(--acid); color: #13231a; }
a { color: var(--accent-dark); font-weight: 700; text-decoration-thickness: 1px; text-underline-offset: .2em; }
a:hover { color: var(--accent); }
button, input, textarea { font: inherit; }
button { border-radius: 0; }
.site-header { border-bottom: 3px solid var(--line); background: var(--header); }
.site-header > .header-actions, .reader { width: min(1100px, calc(100% - 2.5rem)); margin: 0 auto; }
.site-header > .header-actions { align-items: stretch; display: flex; justify-content: space-between; min-height: 78px; }
.brand {
  align-items: center;
  color: var(--ink);
  display: flex;
  font-family: var(--font-display);
  font-size: clamp(1.5rem, 4vw, 2.35rem);
  font-weight: 900;
  letter-spacing: -.045em;
  line-height: .9;
  padding-right: 1.25rem;
  text-decoration: none;
}
.brand::after { background: var(--acid); content: ""; height: .38em; margin-left: .45rem; width: .38em; }
.header-actions { align-items: center; display: flex; gap: 1rem; }
.logout { margin: 0; }
.logout button {
  background: transparent;
  border: 0;
  border-left: 1px solid var(--line);
  color: var(--ink);
  cursor: pointer;
  font-family: var(--font-mono);
  font-size: .72rem;
  font-weight: 700;
  height: 100%;
  letter-spacing: .08em;
  min-height: 44px;
  padding: .75rem 1rem;
  text-transform: uppercase;
}
.logout button:hover, .logout button:focus-visible { background: var(--acid); color: #13231a; }
.reader { padding: 1.5rem 0 5rem; }
.archive-year { font-family: var(--font-display); font-size: 1.5rem; font-weight: 800; list-style: none; margin: 1.5rem 0 .5rem -1.2rem; }
.date-nav {
  align-items: stretch;
  border: 2px solid var(--line);
  display: grid;
  grid-template-columns: 1fr auto 1fr;
  margin-bottom: .75rem;
  min-height: 52px;
}
.date-nav > * { align-items: center; display: flex; min-width: 0; padding: .75rem 1rem; }
.date-nav a { font-family: var(--font-mono); font-size: .72rem; letter-spacing: .04em; text-decoration: none; text-transform: uppercase; }
.date-nav a:hover, .date-nav a:focus-visible { background: var(--acid); color: #13231a; outline: 0; }
.date-nav a:last-child { justify-content: flex-end; text-align: right; }
.date-nav time {
  background: var(--ink);
  color: var(--paper);
  font-family: var(--font-mono);
  font-size: .76rem;
  font-weight: 800;
  letter-spacing: .08em;
  text-align: center;
  text-transform: uppercase;
}
.report-status {
  background: var(--acid);
  border: 2px solid var(--line);
  color: #13231a;
  font-family: var(--font-mono);
  font-size: .75rem;
  font-weight: 700;
  letter-spacing: .02em;
  margin-bottom: .75rem;
  padding: .7rem 1rem;
}
.report-status p { margin: 0; }
.report-status a { color: #13231a; font-weight: 900; margin-left: .4rem; }
.retrospective-notice {
  background: var(--card);
  border: 2px solid var(--line);
  margin-bottom: 1rem;
  padding: 1rem 1.25rem;
}
.retrospective-notice h2 { font-size: 1.15rem; margin: 0 0 .5rem; }
.retrospective-notice p { margin: .35rem 0; }
.report-tools {
  align-items: start;
  border-bottom: 3px solid var(--line);
  border-top: 1px solid var(--line);
  display: flex;
  gap: 1.25rem 2rem;
  justify-content: space-between;
  margin-bottom: 2.5rem;
  padding: .65rem 0;
}
.eyebrow, .section-number, .article-source, .cited {
  font-family: var(--font-mono);
  font-weight: 800;
  letter-spacing: .1em;
  text-transform: uppercase;
}
.eyebrow { color: var(--accent-dark); font-size: .72rem; }
h1, h2, h3, p { margin-top: 0; }
h1, h2, h3 { font-family: var(--font-display); overflow-wrap: anywhere; }
h1 { font-size: clamp(2.7rem, 8vw, 5.8rem); letter-spacing: -.055em; line-height: .92; margin-bottom: 1.25rem; }
h2 { font-size: clamp(2rem, 5vw, 3.65rem); letter-spacing: -.045em; line-height: .98; margin-bottom: 1rem; max-width: 18ch; }
h3 { font-size: 1.2rem; line-height: 1.25; }
.report-meta {
  color: var(--ink-soft);
  display: flex;
  flex-wrap: wrap;
  font-family: var(--font-mono);
  font-size: .72rem;
  gap: .75rem 1.5rem;
  letter-spacing: .035em;
  text-transform: uppercase;
}
.report-feedback { max-width: 420px; width: 100%; }
.report-tools .feedback-control { margin-top: 0; padding-block: 0; }
.video-digest { background: var(--card); border: 3px solid var(--line); box-shadow: 8px 8px 0 var(--acid); margin: 1.5rem 8px 2.5rem 0; padding: clamp(1rem, 3vw, 1.75rem); }
.video-digest-head { align-items: end; display: flex; flex-wrap: wrap; gap: .75rem 1.5rem; justify-content: space-between; }
.video-digest-head h2 { font-size: clamp(1.7rem, 4vw, 2.7rem); margin: 0; }
.edition-selector { display: flex; flex-wrap: wrap; gap: .4rem; }
.edition-selector a { border: 1px solid var(--line); font-family: var(--font-mono); font-size: .68rem; padding: .45rem .65rem; text-decoration: none; text-transform: uppercase; }
.edition-selector a[aria-current="true"] { background: var(--ink); color: var(--paper); }
.video-digest video { background: #000; display: block; margin-top: 1rem; max-height: 70vh; width: 100%; }
.video-links { align-items: baseline; display: flex; flex-wrap: wrap; gap: .5rem 1rem; margin: .7rem 0 0; }
.subtitle-status { color: var(--muted); font-family: var(--font-mono); font-size: .72rem; }
.transcript { border-top: 2px solid var(--line); margin-top: 1.5rem; padding-top: 1.25rem; }
.transcript h3 { font-size: 1.45rem; }
.transcript ol { margin: 0; padding-left: 1.5rem; }
.transcript li { padding: .85rem 0 .85rem .35rem; }
.transcript li + li { border-top: 1px solid var(--line-soft); }
.transcript article > p { max-width: 72ch; }
.transcript .feedback-control { max-width: 520px; }
.event-section { border-top: 1px solid var(--line); margin-top: 1.75rem; padding-top: .5rem; }
.event-disclosure > summary { cursor: pointer; }
.event-disclosure > summary, .articles > summary, .worth-knowing > summary {
  align-items: center;
  display: flex;
  gap: .75rem;
  justify-content: space-between;
  min-height: 44px;
  padding: .65rem .55rem;
  transition: background-color 120ms ease, box-shadow 120ms ease, color 120ms ease;
}
.event-disclosure > summary:hover, .articles > summary:hover, .worth-knowing > summary:hover {
  background: var(--acid);
  box-shadow: inset 5px 0 0 var(--accent);
  color: #13231a;
}
.event-disclosure > summary:focus-visible { outline: 2px solid var(--accent); outline-offset: .25rem; }
.event-disclosure > summary h3 { display: inline; }
.event-article-count { color: var(--muted); font-family: var(--font-mono); font-size: .7rem; margin-left: .75rem; text-transform: uppercase; white-space: nowrap; }
.event-disclosure[open] > summary { margin-bottom: 1.5rem; }
.story-section {
  background: color-mix(in srgb, var(--paper) 92%, transparent);
  border-bottom: 1px solid var(--line);
  border-top: 4px solid var(--line);
  margin: 0;
  padding: clamp(2rem, 6vw, 4.5rem) clamp(.25rem, 3vw, 2rem);
  position: relative;
}
.story-section + .story-section { border-top-width: 1px; }
.section-number { color: var(--accent-dark); font-size: .72rem; margin-bottom: .65rem; }
.section-number::before { background: var(--acid); content: ""; display: inline-block; height: .75em; margin-right: .55rem; width: .75em; }
.consequence { border-left: 4px solid var(--line); color: var(--ink-soft); font-size: .96rem; line-height: 1.6; margin-bottom: 0; max-width: 72ch; padding-left: 1rem; }
.cited { color: var(--accent-dark); font-size: .68rem; }
.worth-knowing { border-bottom: 3px solid var(--line); border-top: 3px solid var(--line); margin-top: 3rem; }
.worth-knowing > summary { cursor: pointer; }
.worth-knowing > summary:focus-visible { outline: 2px solid var(--accent); outline-offset: .25rem; }
.worth-knowing > summary .wk-title { color: var(--ink); font-family: var(--font-mono); font-size: .78rem; font-weight: 900; letter-spacing: .12em; text-transform: uppercase; }
.worth-knowing > summary .event-article-count { margin-left: .75rem; }
.worth-knowing[open] > summary { margin-bottom: 1.5rem; }
.worth-knowing .story-section:first-of-type { margin-top: 0; }
.summary { font-size: clamp(1.05rem, 2vw, 1.18rem); line-height: 1.72; max-width: 72ch; }
.facts { display: grid; gap: 1.5rem; grid-template-columns: repeat(2, minmax(0, 1fr)); margin: 2rem 0; }
.fact { border-top: 3px solid var(--ink); padding-top: .85rem; }
.fact h3 { font-size: 1rem; }
.key-points { font-size: 1.1rem; }
.fact ul { line-height: 1.55; margin-bottom: 0; padding-left: 1.2rem; }
.fact li + li { margin-top: .55rem; }
.assessment { background: var(--assessment); border-left: 6px solid var(--accent); margin: 1.75rem 0; padding: 1rem 1.2rem; }
.assessment h3 { font-family: var(--font-mono); font-size: .75rem; letter-spacing: .08em; text-transform: uppercase; }
.assessment p { color: var(--assessment-ink); line-height: 1.55; margin-bottom: 0; }
.uncertainty { border-bottom: 1px dashed var(--line-soft); border-top: 1px dashed var(--line-soft); padding: .8rem 0; }
.articles { border-top: 1px solid var(--line); margin-top: 2rem; }
.articles > summary { cursor: pointer; }
.articles > summary:focus-visible { outline: 2px solid var(--accent); outline-offset: .25rem; }
.articles > summary { font-family: var(--font-mono); font-size: .72rem; font-weight: 800; letter-spacing: .06em; text-transform: uppercase; }
.article-card { display: grid; gap: 1.25rem; grid-template-columns: minmax(0, 1fr) minmax(250px, .62fr); padding: 1.4rem .55rem; }
.article-card + .article-card { border-top: 1px solid var(--line); }
.article-card h3 { margin-bottom: .5rem; }
.article-source { color: var(--muted); font-size: .68rem; margin-bottom: .35rem; }
.sentiment { color: var(--muted); font-family: var(--font-mono); font-size: .7rem; }
.feedback-control {
  background: var(--annotation);
  border-left: 2px solid var(--line-soft);
  font-family: var(--font-reading);
  margin-top: .75rem;
  padding: .55rem .75rem;
}
.feedback-state { align-items: baseline; display: flex; flex-wrap: wrap; gap: .4rem .75rem; margin-bottom: .55rem; }
.feedback-state strong { font-family: var(--font-mono); font-size: .74rem; text-transform: uppercase; }
.feedback-state small { color: var(--muted); }
.feedback-control details { padding-top: .2rem; }
.feedback-control summary { color: var(--ink-soft); cursor: pointer; font-family: var(--font-mono); font-size: .72rem; font-weight: 750; list-style-position: inside; min-height: 44px; padding: .65rem 0; }
.feedback-form { margin-top: .8rem; }
.feedback-form.htmx-request { opacity: .7; }
.research-flag { background: var(--annotation); border: 2px solid var(--line); box-shadow: 6px 6px 0 var(--acid); color: var(--ink); margin: 1.4rem 6px 1.4rem 0; padding: 1rem; }
.research-head { align-items: baseline; display: flex; gap: .6rem; margin: 0; }
.research-head strong { color: var(--accent-dark); font-family: var(--font-mono); font-size: .78rem; letter-spacing: .06em; text-transform: uppercase; }
.research-question { font-style: italic; margin: .5rem 0 .7rem; }
.research-flag fieldset { border: 1px solid var(--line); margin: .6rem 0; padding: .6rem .8rem; }
.research-flag legend { font-family: var(--font-mono); font-size: .72rem; font-weight: 700; padding: 0 .3rem; }
.research-flag fieldset[name="flag_verdict"] { display: grid; gap: .65rem; grid-template-columns: repeat(2, minmax(0, 1fr)); }
.verdict-option {
  align-items: center;
  background: var(--control);
  border: 2px solid var(--line);
  cursor: pointer;
  display: flex;
  font-family: var(--font-mono);
  font-size: .75rem;
  font-weight: 900;
  gap: .65rem;
  justify-content: center;
  letter-spacing: .08em;
  min-height: 48px;
  padding: .65rem 1rem;
  text-transform: uppercase;
  transition: background-color 120ms ease, box-shadow 120ms ease, color 120ms ease, transform 120ms ease;
}
.verdict-option:hover { background: var(--acid); color: var(--shadow); transform: translate(-2px, -2px); }
.verdict-option:has(input:checked) { background: var(--ink); box-shadow: 4px 4px 0 var(--acid); color: var(--paper); }
.verdict-option:has(input:focus-visible) { outline: 3px solid var(--accent); outline-offset: 3px; }
.verdict-option input, .reason-option input { accent-color: var(--accent); }
.verdict-option input:checked { accent-color: var(--acid); }
.reason-option { align-items: center; display: flex; gap: .5rem; min-height: 44px; padding: .3rem 0; }
.research-flag textarea { margin: .6rem 0; }
.rating-actions { display: grid; gap: .65rem; grid-template-columns: repeat(3, 1fr); }
.rating-actions button, .research-flag button { border: 2px solid var(--ink); cursor: pointer; font-family: var(--font-mono); font-size: .72rem; font-weight: 800; letter-spacing: .04em; min-height: 44px; padding: .65rem 1rem; text-transform: uppercase; }
.rating-actions button[value="positive"] { background: var(--positive); color: var(--button-ink); }
.rating-actions button[value="negative"] { background: var(--negative); color: var(--button-ink); }
.rating-actions button[value=""] { background: var(--control); color: var(--ink); }
.rating-actions button:hover, .rating-actions button:focus-visible, .research-flag button:hover, .research-flag button:focus-visible { box-shadow: 3px 3px 0 var(--shadow); transform: translate(-2px, -2px); }
.note-label { display: block; font-family: var(--font-mono); font-size: .7rem; font-weight: 700; margin: .8rem 0 .35rem; text-transform: uppercase; }
.feedback-form textarea { background: var(--control); color: var(--ink); border: 1px solid var(--control-line); display: block; min-height: 86px; padding: .7rem; resize: vertical; width: 100%; }
.empty { border: 3px solid var(--line); box-shadow: 10px 10px 0 var(--accent); margin: 12vh auto; max-width: 680px; padding: clamp(2rem, 7vw, 4rem); text-align: left; width: calc(100% - 2.5rem); }
.empty h1 { font-size: clamp(2.5rem, 8vw, 5rem); }
.empty .report-status { margin-top: 1.5rem; }
.login-shell { display: grid; min-height: 100vh; padding: 1.25rem; place-items: center; }
.login-card { background: var(--card); border: 3px solid var(--line); box-shadow: 10px 10px 0 var(--acid), 13px 13px 0 var(--line); max-width: 520px; padding: clamp(1.75rem, 6vw, 3.5rem); width: 100%; }
.login-card h1 { font-size: clamp(3rem, 11vw, 5.5rem); }
.login-card label { font-family: var(--font-mono); font-size: .72rem; font-weight: 800; letter-spacing: .06em; text-transform: uppercase; }
.login-card input { background: var(--control); color: var(--ink); border: 2px solid var(--control-line); margin: .5rem 0 1rem; min-height: 48px; padding: .75rem; width: 100%; }
.login-card input:focus { border-color: var(--ink); box-shadow: 4px 4px 0 var(--acid); outline: 0; }
.login-card button { background: var(--ink); border: 2px solid var(--ink); color: var(--button-ink); cursor: pointer; font-family: var(--font-mono); font-size: .75rem; font-weight: 800; letter-spacing: .08em; min-height: 48px; padding: .7rem 1.2rem; text-transform: uppercase; width: 100%; }
.login-card button:hover, .login-card button:focus-visible { background: var(--acid); color: #13231a; }
.error { color: var(--negative); font-weight: 800; }
@media (prefers-color-scheme: dark) {
  :root {
    --ink: #ece5d1;
    --ink-soft: #c7c0ab;
    --muted: #aaa48f;
    --paper: #171714;
    --card: #211f19;
    --line: #d2cbb7;
    --line-soft: #554f42;
    --acid: #b9dd38;
    --accent: #ef7048;
    --accent-dark: #ff9270;
    --positive: #3d7454;
    --negative: #c84f36;
    --header: rgba(23,23,20,.96);
    --assessment: #2b2922;
    --assessment-ink: #d0c8b5;
    --annotation: #24231e;
    --control: #2a2821;
    --control-line: #746d5e;
    --button-ink: #fffaf0;
    --grid-line: rgba(236,229,209,.045);
    --shadow: #000;
  }
}
@media (max-width: 720px) {
  .site-header > .header-actions, .reader { width: min(100% - 1.25rem, 1100px); }
  .site-header > .header-actions { min-height: 64px; }
  .brand { font-size: 1.55rem; }
  .reader { padding-top: .75rem; }
  .date-nav { grid-template-columns: 1fr 1fr; }
  .date-nav time { grid-column: 1 / -1; grid-row: 1; text-align: center; }
  .date-nav a { grid-row: 2; }
  .facts, .article-card { grid-template-columns: 1fr; }
  .report-tools { display: block; }
  .report-feedback { margin-top: .75rem; max-width: none; }
  .video-digest { margin-right: 0; }
  .video-digest-head { align-items: start; display: block; }
  .edition-selector { margin-top: .75rem; }
  .rating-actions { grid-template-columns: 1fr; }
  .article-card { gap: .4rem; }
  .story-section { padding-left: .35rem; padding-right: .35rem; }
  .event-disclosure > summary, .articles > summary, .worth-knowing > summary { align-items: flex-start; }
  h1 { font-size: clamp(2.5rem, 12vw, 4.4rem); }
  h2 { font-size: clamp(2rem, 10vw, 3.25rem); }
}
@media (max-width: 420px) {
  .brand::after { display: none; }
  .logout button { padding-inline: .7rem; }
  .date-nav > * { padding-inline: .65rem; }
  .event-disclosure > summary { display: block; }
  .event-article-count { display: block; margin: .3rem 0 0; }
  .worth-knowing > summary .event-article-count { margin-left: 0; }
  .research-flag fieldset[name="flag_verdict"] { grid-template-columns: 1fr; }
  .empty { margin-top: 7vh; width: calc(100% - 1.25rem); }
}
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { scroll-behavior: auto !important; transition-duration: .01ms !important; }
}
"""
