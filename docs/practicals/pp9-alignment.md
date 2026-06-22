
# Programming Practical 9: Alignment

This Programming Practical makes **preference optimization mechanically transparent**. Instead of building a full RLHF stack, you implement the *core* of two alignment methods — **DPO** and **REINFORCE** — as three short functions, and then *watch the log-probabilities move* as the model learns.

It is the hands-on companion to lecture **section 6** (Policy Gradient → PPO → GRPO) and **section 7** (DPO & Best-of-N).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/paulnovello/Advanced-AI/blob/main/PP9%3A%20Alignment/dpo_reinforce.ipynb)

Solution: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/paulnovello/Advanced-AI/blob/main/PP9%3A%20Alignment/dpo_reinforce_solutions.ipynb)

Everything is kept tiny on purpose so you can read the actual numbers:

- a **toy reward model** — *be enthusiastic!* — that simply counts `!` and upbeat words, so the optimization target is obvious;
- a **tiny model** (`SmolLM2-135M-Instruct`), small enough to actually train on a free Colab CPU or T4.

!!! Note
    Optional but smoother: `Runtime → Change runtime type → T4 GPU`. CPU also works — sampling is just slower. The training cells are the heavy part.

---