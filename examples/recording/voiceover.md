# YC Demo Voiceover (Terminal + Cards)

## Cards Segment (0:00-0:30)

### Card 1
Coding and vision benchmarks are saturating, but frontier multimodal models are still weak at hearing.

### Card 2
Most audio evals are too small and too narrow. They miss the failure modes that matter in production.

### Card 3
Phonon is building the audio equivalent of MMLU: hard, human-labeled benchmark datasets designed to stay unsaturated.

### Card 4
`audiobench` is our open wedge. The harness drives adoption, and proprietary benchmark datasets become the paid layer.

### Card 5
Now I will run the live benchmark flow end to end.

## Terminal Segment (0:30-2:45)

I warm up the model cache to remove setup friction.  
Then I list suites and inspect `ab/asr-robust`.

Next I run a baseline model on focused conditions: clean audio, phone-band audio, and reverb.  
The key signal is not one number, it is the per-condition failure map and delta versus clean.

Now I run a stronger candidate model with the same setup.  
The compare command shows exactly which model wins each condition and by how much.

Every run emits a deterministic artifact with a hash, so results are reproducible and auditable.  
Finally, push emits a signed payload that can feed leaderboard or internal eval pipelines.

Phonon is building benchmark infrastructure for hearing-capable multimodal AI.
