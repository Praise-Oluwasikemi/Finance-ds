# Learning Log

One short entry per session, newest at the top. Keep it honest. This is your evidence of progress, and good material for interview stories.

<!-- Copy this template for each session -->
## Session 1: Setup + 00_start_here.ipynb · 2026-10-23

**What I did:** 
Installed GitHub Desktop, cloned Finance-ds, created the finance-ds conda 
environment, opened Jupyter, ran 00_start_here.ipynb, fixed the compound 
interest bug, committed and pushed to GitHub.

**What clicked:** 
Commit vs Push — commit saves on my laptop, push sends to GitHub. Also 
that a bug can run without an error message; the logic is wrong, not the 
syntax, and the check() box is what catches it.

**What was hard / where I got stuck:** 
The conda environment didn't exist on first try (got 
EnvironmentNameNotFound). Re-ran the conda create command and it worked. 
Also couldn't find the Push button at first — used git push from Anaconda 
Prompt to confirm everything was up to date.

**AI help used:** 
hint and concept explanation — for the compound interest formula, and for 
troubleshooting conda/GitHub Desktop steps.

**Peeked at a solution?** 
no

**My answer to a ✍️ question I'm proud of:** 
Fixed the compound interest formula: 
principal * (1 + annual_rate / periods_per_year) ** (periods_per_year * years) 
— understood why it's ** (n*t) and not * years.

**One thing I can do now that I couldn't before:** 
Run a full session end to end: activate environment → open Jupyter → write 
and check code → commit and push to GitHub.

---
