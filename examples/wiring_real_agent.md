# Wiring a real agent

`agent-evals` evaluates anything that can answer with recorded turns. The
contract is small:

```python
# my_agent.py
from agent_evals.case import Case, Turn

def build_agent():
    # Return a callable: (Case) -> list[Turn], one Turn per prompt/answer cycle.
    def agent(case):
        turn = my_real_agent_answer(case.prompt)
        return [Turn(prompt=case.prompt, response=turn.text, tool_calls=turn.calls)]
    return agent
```

Run it:

```bash
PYTHONPATH=src python3 -m agent_evals.cli cases.json --agent my_agent.py:build_agent --threshold 0.9
```

Every grader sees only what you hand it: the response text and the tool calls
(`name`, `args`, `result`). No other side channel exists — the graders decide
from the transcript, which is what makes the verdict reproducible.

## Record once, replay forever

Capture the turns your agent produced for each case and store them in the case
file. From then on, CI evaluates the recorded transcripts (`replay` mode):

```bash
PYTHONPATH=src python3 -m agent_evals.cli cases.json --threshold 0.9
```

Replay is:

- **Sealed**: no agent runs, no model calls, no network — a case suite is just data.
- **Cheap**: the whole suite runs in milliseconds, so it is safe to gate every PR.
- **Fair**: behavior is only compared once, frozen at evaluation time, so it cannot drift under CI while the code moves.

Keep `agent` mode in the loop for parity: run it on a schedule or before
promoting prompts and verify the live agent still reproduces the recorded
turns (`examples/support_agent/cases.json` + `:build_agent` demonstrates the
contract — the example test asserts byte-for-byte reproduction).
