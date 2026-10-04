# Working rules for Claude

## Process: waterfall

Work in phases, in this order. Never start a phase until I have approved the previous one in chat.

1. **Requirements**: interview me and write down what we are solving, the constraints, and how success is measured.
2. **Design**: propose the approach (files, scripts, parameters, experiments). No code yet.
3. **Implementation**: write only what the approved design says.
4. **Verification**: run the agreed checks and report the results as they are, including failures.
5. **Delivery**: update `data.md` / `README.md`, then commit only when I ask.

At the end of each phase, summarise it and stop to wait for my approval.

## Interview first, don't act on your own

- Before any task, interview me with questions until you have enough context: goal, data, constraints, compute, time I can spend, and how we measure success.
- Look at the code and data to prepare good questions, but don't change anything while you do.
- Don't make decisions for me. If something is unclear or there are several options, ask; don't pick a default and carry on.
- Don't write, delete, move, retrain, re-split, or sync data unless I approved that specific step.
- If my answers contradict each other or contradict the data, point it out and ask.
- If something comes up mid-phase that wasn't in the approved plan, stop and ask.
