# Local GUI

`audiobench --gui` launches a small local Gradio app that wraps the same Python
APIs the CLI uses. It is optional and aimed at one thing: making it easier to
**structure testing sessions** (multi-suite, multi-model matrices) and to
**browse results** organised by session. Everything in the GUI can be done from
the terminal as well — the GUI is not a replacement for the CLI.

## Install and launch

```bash
pip install -e ".[gui]"
audiobench --gui
```

The app binds to `http://127.0.0.1:7861` by default and opens your browser.
Flags:

- `--gui-port 7862` — pick a different port
- `--gui-host 0.0.0.0` — expose to your LAN (default is localhost only)
- `--no-open` — don't auto-open a browser

## Test Builder

The Test Builder composes a `matrix.yaml` — the same format consumed by
`audiobench run-matrix --matrix matrix.yaml`.

- **Cells table** — one row per `(suite, model)` cell. Columns: `name`,
  `suite`, `model`, `seed`, `limit`, `conditions`, `pack`, `profile`. Use the
  suite/model dropdowns + **Add cell** to append rows; the table itself is
  editable for quick tweaks.
- **Gate thresholds (optional)** — pass/fail caps that get applied to each
  cell after it runs, the same checks `audiobench gate` performs. They turn a
  run into a CI-style green/red decision: e.g. fail if
  `sound_id.min_weighted_recall < 0.6` or `asr_robust.max_weighted_mean_wer > 30`.
  Leave every field blank to skip gating; fill any field and the YAML preview
  gains a `gate:` section.
- **matrix.yaml preview** — live YAML on the right. **Save** writes it to
  disk; **Load** rehydrates the builder from an existing YAML or JSON file.
- **Run matrix** — executes every cell in-process (same code path as
  `audiobench run-matrix`), writes per-cell run JSONs to `output_dir`, and
  drops a `summary.json` alongside them. The Results tab will pick it up.

## Results

A "testing session" is any directory under `results/` that contains a
`summary.json` (the file `run-matrix` writes). The Results tab lists those,
plus ad-hoc top-level `results/*.json` runs.

- Click a session row to see its per-cell summary table (the same one
  `audiobench run-matrix` prints).
- Click a cell row to load that run's headline summary
  (`audiobench inspect`-style detail can be opened from the **Inspect** tab).

You can point the **results root** input at any directory if your runs live
somewhere other than `./results`.

## Secondary tabs

These are thin forms over the existing CLI commands and each tab shows the
equivalent terminal command so you can copy it if you prefer the shell:

- **Run** — `audiobench run <suite> --model <id> ...`
- **Compare** — `audiobench compare A.json B.json`
- **Inspect** — `audiobench inspect run.json --clip N` / `--mixture N`
- **Gate** — `audiobench gate run.json --thresholds gate.yaml`
- **Push** — `audiobench push run.json [--repo ...] [--dry-run]`

Nothing in the GUI is privileged: any artifact it writes (`matrix.yaml`,
per-cell run JSONs, `summary.json`) is interchangeable with the CLI's output.
