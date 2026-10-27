# Silt

Silt — движок грязи и бездорожья. Этот срез — короткий замкнутый грунтовый круг: несколько луж, брустверы, колея, которая остаётся, и полноприводный багги. Камера по умолчанию — фиксированный изометрический вид на весь круг. Колёса продавливают колею, след темнеет, грязь липнет к шинам и панелям, а глубокая жижа на малой скорости сажает машину.

The solver is ours. Soil is a heightfield, not a stock vehicle package and not a 3D SPH sim. Chassis collision only keeps the body from diving through the world; the forces in the mud come from Bekker, Janosi–Hanamoto and a Bingham-like drag.

## Build and run

Rust 1.85 or newer (the repo pins 1.99.1 via `rust-toolchain.toml` when rustup is installed).

```bash
cargo run --release
```

That opens a window. On a machine with no discrete GPU, lavapipe can drive an offscreen capture. `--software` points Vulkan at lavapipe when that ICD is installed and clears `DISPLAY` so a broken X server is not picked up:

```bash
cargo run --release -- capture --software \
  --width 1280 --height 720 --fps 30 --seconds 12 \
  --video mud-demo.mp4 \
  --still-start mud-demo-start.png \
  --still-puddle mud-demo-puddle.png \
  --still-ruts mud-demo-ruts.png
```

`--stills-only` jumps the same scripted drive and writes just the three stills. The script is deterministic: idle on the gate, then one lap that crosses a puddle, crests a berm, and leaves the rut in view. Tests use the same inputs. The capture camera is the fixed isometric frame.

```bash
cargo test
```

`scripted_drive` checks that the lap cuts a deeper rut, puts mud on the body, throws splash, and slows in the soft water. Stopping in the infield pond and spinning digs the buggy in.

## Controls

| Key | Action |
| --- | --- |
| W / Up | Throttle |
| S / Down | Brake, then reverse once the buggy is slow |
| A / Left, D / Right | Steer |
| V | Switch which vehicle the pedals drive. The other one keeps riding the lap |
| C | Toggle the chase camera. The default is a fixed isometric view of the loop. Chase follows whichever vehicle V selected |
| Q / E | Orbit the chase camera |
| R | Replay the recorded lap from the side, top, bumper, and rear three-quarter |
| T / F | Raise / lower the chase camera |
| Esc | Quit |

The isometric camera stays put so the whole loop, the puddles and the rut read at once. The chase camera follows the vehicle selected with V. R replays the first closed lap from a fresh start, walking the side, top, bumper, and rear three-quarter cameras.

Opening the window or the page plays that lap with both vehicles. The first drive key (W, S, A, D, or the arrows) takes the selected one. The other keeps the lap, so the track still has two machines. The capture is the same lap, with nobody at the wheel.

## Браузер

Та же симуляция открывается в браузере: WebGPU, а если его нет на машине — WebGL2. Решатель тот же. Камера по умолчанию — изометрия. Пока не нажата педаль, багги и мотоцикл сами едут круг. Первая клавиша газа, тормоза или руля забирает выбранную машину; вторая продолжает круг. V переключает, кого ведёте. Скорость на западной прямой (справа в этом виде) по-прежнему срывает машину с гребня: кувырок, кратер и комья грязи, которые летят дугой и садятся на поле высот. Сборка для GitHub Pages лежит в `.github/workflows/pages.yml`. Адрес https://x3-team.github.io/Dirty-gen/ открывается после того, как в настройках репозитория включён Pages и источник — GitHub Actions.

Нужен [trunk](https://trunkrs.dev/). Из корня репозитория:

```bash
trunk serve --release
```

Страница: http://127.0.0.1:8080 . Сборка без `--release` слишком медленная, чтобы набрать скорость на гребень. Кликните по сцене, иначе клавиши уходят странице.

| Клавиша | Действие |
| --- | --- |
| W / ↑ | газ |
| S / ↓ | тормоз, затем задний ход |
| A / ←, D / → | руль |
| V | кого ведёте: багги или кроссовый мотоцикл. Вторая машина продолжает круг |
| C | камера погони. По умолчанию изометрия. Погоня следует за выбранной машиной |
| Q / E | орбита камеры погони |
| R | повтор записанного круга: сбоку, сверху, с бампера, затем сзади в три четверти |
| T / F | выше / ниже |
| Esc | выход |

Адрес с `?gl` принудительно включает WebGL2.

## What the soil does

The field is a grid (40 m, 192 cells on a side). Each cell stores height, loose mud, wetness and water depth.

- **Bekker** pressure–sinkage turns wheel load into how far the tire sinks. Wetter, looser ground has a smaller modulus, so the same weight sinks further.
- **Janosi–Hanamoto** shear is the traction. It saturates with slip. On firm ground a simple tire curve still does most of the work; once the rut is deep, the soil term takes over.
- **Slip-sinkage** digs a plastic rut. A rolling tire leaves that groove in the heightfield after it has passed — a pair of darker floors with the cut soil piled as a lighter berm on each shoulder. Spinning in a puddle digs deeper. The rut stays. It is the displaced mesh, not a decal.
- **Bingham-like drag** is why low speed bogs. Fresh mud resists, but a spinning tire can still creep. Viscosity is what kills speed when you hit the water with momentum. If you stop in the deep part and keep the throttle in, the tire digs a hole and the lip of that hole holds the buggy. Reverse while you are still rolling and you can leave.
- **Tracks** darken the albedo and change roughness along the contact patch. That is stored in the field, not a decal. The groove in the picture is the heightfield mesh the stamp actually lowered, and the berm it piled beside the cut. After the buggy has passed, one groove runs along each side of the path it drove: a darker floor a few centimetres down, and a lighter berm of the soil pushed aside. The tires cut deeper where they slip in the water. That is the depth the camera shows. Carrying speed over the west-straight crest launches the buggy, rotates it in the air, and the landing digs a crater and throws the live splash. The splash is a few large mud clumps with volume, plus smaller droplets. They arc, then sit in the crater as dark glossy mud, not stones. Flat cards are not drawn. The rut floor and the soil around a puddle carry a wet sheen. The water is dark muddy brown in the middle and lighter muddy brown at the shore, with a quiet reflection. The same crest does it for the player. No key.
- **Shallow water** is a small pipe scheme. The puddle wets the cells under it. A wheel throws that water forward and raises splash particles.
- **Adhesion** is an ODE on each tire and on a handful of body panels (nose, sides, belly, fenders, and so on). Pickup grows with slip and speed, mass drips off, and the panel state drives the mud mask: clumps and streaks, not one flat brown multiply.

The buggy is a short-travel 4x4: four wheels, a spring matched to the Bekker load, equal drive torque, steer on the front axle. The picture is a procedural open-wheel frame — cage, seats, arms and coil springs that shorten with the spring travel and hang at droop in the air. It is not a multibody A-arm model.

A motocross bike runs the same loop, just inside the buggy's line. It is the same soil — Bekker, Janosi–Hanamoto, the wheel stamp, and the crest launch — on a lighter chassis with a front fork and a rear shock. Those shorten when the tire loads and hang at droop in the air. The crest flips it and the landing throws the same wet clumps. V hands it the pedals. The isometric camera stays the default.

`.github/workflows/pages.yml` builds that page with `trunk build --release`. https://x3-team.github.io/Dirty-gen/ is the project address once Pages is enabled and the source is GitHub Actions.

## Not in this slice

A race mode, a track editor, menus, multiplayer, and a large world are intentionally absent. The field is one short loop — a few puddles, low berms, and the ground around them — so the soil model can be seen running in real time.

## Layout

- `src/soil.rs` — heightfield, Bekker, Janosi, water, rut stamp
- `src/vehicle.rs` — buggy, suspension, tire / terramechanics blend, mud ODE
- `src/particles.rs` — splash, clods, drips
- `src/world.rs` — coupling and the scripted demo
- `src/render.rs`, `src/shader.wgsl`, `src/mesh.rs` — wgpu forward renderer
