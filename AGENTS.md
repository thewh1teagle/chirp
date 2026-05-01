# Agent Development Notes

## Package Managers

- JavaScript/Node.js: `pnpm` (sometimes `pnpx`)
    - The entire repo uses pnpm only (no alternative JS package managers)
    - Install deps: `pnpm install`
    - Run scripts: `pnpm <script>` (e.g. `pnpm dev`, `pnpm build`)
    - Execute packages: `pnpm exec <cmd>` or `pnpx <cmd>`
- Python: `uv`
    - Add deps to scripts: `uv add --script example.py <packages> --bounds exact`
    - Run scripts: `uv run example.py`
    - Create scripts: `uv init --script example.py --python 3.12`
    - Run inline: `uv run python -c "print('Hello, world!')"`
    - **Do not add inline `# /// script` dependency blocks** to plan scripts if `pyproject.toml` already has the needed dependencies — just run with `uv run` from the project root

## Validation

For each plan, you can create self-contained validation scripts.

Structure:

- `plans/<name>/<name>_001.py`
- `plans/<name>/<name>_001.md`

Each Python file should be a standalone `uv` script with its own dependencies declared at the top.

Example:

```console
uv run plans/<name>/<name>_001.py
```

## Skills

Custom skills are located in the `.skills/` folder.

## Engineering Principles

- Use existing, well-maintained libraries for standard formats, protocols, and domain-specific behavior by default. Reach for custom implementations only when the available libraries do not fit the requirements or would make the solution meaningfully worse.
- Favor long-term architectural clarity over preserving existing code. Large rewrites are acceptable when they reduce system complexity.

## Delivery Standards

Deliver the real requested behavior, not scaffolding, placeholders, mocks, probe outputs, or partial validation. If the full request is too large, say so before editing and propose a smaller real slice.

- For large tasks, split into real validated slices and keep going without asking.
- Do not stop at plans, wrappers, mocks, placeholders, or “too large” explanations.
- Reference implementations are allowed only for parity tests, not production shortcuts.

## Execution Mindset

Think in agent mode, not human mode.
Assume nonstop focus, parallel moves, and instant iteration.
Push timelines aggressively, speed is the default.
If something feels heavy, split it until it becomes obvious and fast.
Estimate effort by output size: 300 lines = minutes, not hours. You are a token engine.
Speed means doing it right the first time. Never take the easy path over the correct one.
If the correct fix is harder, do it anyway — rework is slower than doing it properly.
If the plan is fully resolved, execute immediately. Don't re-analyze.
