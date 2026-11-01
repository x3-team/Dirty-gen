# UE5 comparison bench

Silt stills next to the UE5 fluid-mod screenshots Vitalii picked as the bar,
same scene, similar camera. Silt left, UE5 right. Each scene gets a 1-5
score; the target is 4+ everywhere.

## Run

```sh
cargo build --release
./target/release/silt capture --software --width 1280 --height 720 --bench out/bench
python3 tools/bench_sbs.py out/bench          # needs Pillow
open out/bench/index.html
```

`--software` uses lavapipe (`apt-get install mesa-vulkan-drivers`); drop it on
a machine with a GPU. A full run is about 1.5 minutes on lavapipe after the
build. `--bench-scenes <file>` reads another scene list.

Output in `out/bench`:

- `ours/<id>.png`: the Silt still
- `sbs/NN-<id>.jpg`: 1920x574, Silt left, reference right
- `index.html`: all frames with a score and note field; "Export" copies the
  scores as JSON (they stay in the browser otherwise)
- `bench.json`: scene list and which ones rendered

## CI

`.github/workflows/bench.yml` (`ue5-bench`) runs the same steps on
`workflow_dispatch` and uploads `out/bench` as the `ue5-bench` artifact.
It is manual so it does not add a run to every PR or merge.

## Scenes

`bench/scenes.txt` is the list. One line per scene: preset, where the still
is taken, camera eye and target, reference, title. Cameras are data, so a
shot can be lined up with its reference without a rebuild.

| Scene | Reference | Silt source |
|---|---|---|
| snow-ruts | ue5-96 | snow blanket, ruts, berms (`snow_ruts_world`) |
| snow-circles | ue5-97 | same scene, turn arc from above |
| ford | ue5-104 | `silt::ford` river, Niva wading at 5 s |
| mud | ue5-105 | through-drive, clay slurry ruts |
| mud-overview | video frame | through-drive, rut field overview |
| dust | ue5-93 | through-drive, dry exit with dust |
| river-rocks | ue5-103 | `silt::ford` river, Niva in the channel (`river` preset) |
| beach-swash | ue5-98 | `silt::ford` with a swell on the beach (`beach` preset) |
| beach-rock | ue5-99 | `silt::ford` river, shallows by a rock |
| turbid-water | ue5-100 | missing: no open turbid water yet |
| smoke-column | ue5-89 | missing: no point-source smoke yet |

A `missing` scene renders a card that names what is missing instead of a
drawn substitute. When a solver scene lands (ford & beach, dust & fog),
add a preset in `src/bench.rs` and switch the line from `missing`.

Out of scope: fire, aurora, underwater grass, blood (also in the UE5 set).

## References

`bench/refs/` holds the UE5 screenshots and one frame of the UE5 demo video,
downscaled to 960 px wide. They are comparison material only and come from
Vitalii's uploads; no external links.
