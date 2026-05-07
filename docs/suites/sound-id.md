# `ab/sound-id`

Sound-event identification on mixtures of labeled clips. For each mixture, the model is asked once per candidate label using the bundled prompt set.

```bash
audiobench run ab/sound-id --model heuristic-v0
```

## How it scores

For each `(pack, condition)` row:

- **recall** — of the sounds actually in the mixture, what fraction did the model correctly say "yes" to?
- **precision** — of the times the model said "yes", what fraction were actually present?
- **F1** — combined score blending recall and precision.
- **FPR** — for sounds that are NOT in the mixture (distractors), how often does the model still say "yes"?

The headline is `components understood: X / Y` — across every mixture, X is how many ground-truth components the model identified out of Y total. That's the number meant for a tweet.

## Conditions

Conditions are mixture sizes:

| Condition | N | Notes |
|---|---|---|
| `solo` | 1 | sanity check |
| `pair` | 2 | |
| `triple` | 3 | |
| `quad` | 4 | hardest, polyphony stress test |

Run a subset:

```bash
audiobench run ab/sound-id --model heuristic-v0 --conditions solo,pair
```

## Packs

Each `ab/sound-id` run targets one or more **packs**. Each pack defines a label set and source dataset(s).

| Pack | Source | Labels | License |
|---|---|---|---|
| `demo` | Procedural (bundled, no download) | `siren`, `alarm`, `dog_bark`, `engine`, `glass_breaking`, `baby_cry`, `coughing`, `water`, `vacuum`, `speech` | bundled |
| `core` | FSD50K (single-positive PP filter) | ~80 high-confidence classes from the AudioSet ontology | CC-BY 4.0 / CC0 (user-supplied) |
| `home` | DESED synthetic subset | `alarm_bell`, `cat`, `dishes`, `frying`, `blender`, `water`, `speech`, `vacuum`, `dog`, `electric_shaver` | open (user-supplied) |
| `cabin` | FSD50K + UrbanSound8K | `engine`, `traffic`, `baby_cry`, `music`, `speech`, `car_horn`, `siren`, `drilling` | non-commercial research |
| `security` | UrbanSound8K | `gun_shot`, `siren`, `car_horn`, `dog_bark`, `jackhammer` | non-commercial research |
| `health` | ESC-50 medical subset | `coughing`, `sneezing`, `breathing`, `snoring`, `crying_baby` | non-clinical scope |

The `demo` pack runs with no downloads and powers the headline demo. Other packs require user-supplied data at `~/.cache/audiobench/sound_id/<source>/`.

```bash
audiobench list-packs
audiobench info ab/sound-id
audiobench info ab/sound-id --pack home
```

### Bringing your own data

```text
~/.cache/audiobench/sound_id/
  fsd50k/
    FSD50K.dev_audio/...
  urbansound8k/
    audio/fold1/...
  desed/
    synthetic21_train/soundscapes/...
  esc50/
    audio/...
```

`audiobench info ab/sound-id --pack <name>` prints the expected layout. If files are missing, the suite skips that pack with a clear message rather than failing.

## Mixtures

Three layers, additive.

### Default — canned, seeded mixture set per pack

Zero authoring. The default mixture set is deterministic from the pack and the seed:

```bash
audiobench run ab/sound-id --pack demo --model heuristic-v0
```

### Inline `--mix` — one mixture per flag

`+`-separated labels. Repeatable.

```bash
audiobench run ab/sound-id --mix "siren+glass_breaking+baby_cry" --model heuristic-v0
audiobench run ab/sound-id --mix "engine+baby_cry" --mix "engine+baby_cry+music" --model heuristic-v0
```

### Recipe file (YAML or JSON)

Repeatable scenarios with per-source dB levels and optional pinned source files:

```yaml title="scenarios/factory_floor.yaml"
mixtures:
  - name: factory_alarm
    labels: [siren, glass_breaking]
    snr_db: 0

  - name: cabin_baby_over_engine
    label_levels:
      engine: 0
      baby_cry: -3
      vacuum: -6
```

```bash
audiobench run ab/sound-id --recipes scenarios/factory_floor.yaml --model heuristic-v0
```

When `--mix` or `--recipes` is used, results land under a `custom` condition. The run hash includes the canonicalized mixture spec so any custom run is bit-reproducible.

### Mixture preview

Render a mixture WAV without running probes — useful for demo prep, debugging levels, and authoring recipes:

```bash
audiobench mix preview --labels siren,glass_breaking,baby_cry --output preview.wav
audiobench mix preview --recipes scenarios/factory_floor.yaml --name cabin_baby_over_engine --output cabin.wav
```

## Per-mixture forensic view

```bash
audiobench inspect results/sound-id-heuristic.json --mixture 12
```

```text
mixture 12 (pack=demo, condition=triple)
  ground truth: siren, glass_breaking, dog_bark
  source clips:
    siren           demo://siren@0
    glass_breaking  demo://glass_breaking@0
    dog_bark        demo://dog_bark@0

  model: heuristic-v0
  prompts: version=yesno-v1, parser=v1, ensemble=off (single prompt), source=bundled
  yes responses:
    siren           ✓
    dog_bark        ✓
    glass_breaking  ✗  FALSE NEGATIVE
    chainsaw        ✗  FALSE POSITIVE (distractor)
    car_horn        ✗  (distractor, correct)

  recall    : 2/3 = 0.67
  precision : 2/3 = 0.67
  components understood: 2 of 3
```

When the run was made with `--prompt-ensemble N`, `inspect` also prints a per-paraphrase breakdown showing each rendered prompt and the model's individual yes/no for it.

## See also

- [Prompt protocol](../reference/prompts.md) — versioning, custom prompt files, ensembles.
- [Models](../models/index.md) — adapter list and how to add your own.
- [Reproducibility guarantees](../reference/reproducibility.md) — seeds, hashing, run JSON schema.
