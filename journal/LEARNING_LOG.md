# Learning Log

One short entry per session, newest at the top. Keep it honest. This is your evidence of progress, and good material for interview stories.

<!-- Copy this template for each session -->
## Session 2: 1_finance_maths.ipynb · 2026-10-23

**What I did:**
Finished the finance maths notebook: compound interest, monthly loan
payments, and net present value. Fixed a few bugs on the way.

**What clicked:**
The `raise NotImplementedError` thing is just a "fill me in" marker.
And in Python, where you put `return` matters: inside a loop it stops
early.

**What was hard / where I got stuck:**
- Monthly payment: forgot the zero-interest case, which crashed the
  code. Fixed it.
- NPV: my `return` was inside the loop, so it only counted the first
  cash flow. Moved it out and it worked.

**AI help used:**
hints and explanations: formulas, the crash fix, and spotting the
indentation mistake.

**Peeked at a solution?**
no

**My answer to a ✍️ question I'm proud of:**
NPV tells a business if a project is worth doing after accounting for
time. Positive = good, negative = not worth it.

**One thing I can do now that I couldn't before:**
Write a finance formula in code, test it, and fix it when it breaks.

---

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
