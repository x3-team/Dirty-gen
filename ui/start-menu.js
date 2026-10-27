// Start screen. The oval behind it is the live scene.
// #silt-menu-state is the hook the wasm frame reads once per action.

const state = ensureState();
const canvas = document.getElementById("silt");

const screen = document.createElement("section");
screen.id = "silt-start";
screen.className = "start-screen";
screen.setAttribute("aria-label", "Старт заезда");

const board = document.createElement("div");
board.className = "start-board";

const mark = document.createElement("p");
mark.className = "mark";
mark.textContent = "Силт";

const buggy = rideButton("buggy", "Багги", "Багги");
const bike = rideButton("bike", "Мото", "Мотоцикл");
const start = actionButton("act act-start", "Старт", "start");
const restart = actionButton("act act-restart", "Заново", "restart");

const note = document.createElement("p");
note.className = "note";
note.textContent = "Вторая машина едет круг.";

board.append(mark, buggy, bike, start, restart, note);
screen.append(board);
document.body.append(screen);

const raceRestart = actionButton("act act-restart race-restart", "Заново", "restart");
raceRestart.hidden = true;
document.body.append(raceRestart);

let ride = state.dataset.vehicle === "bike" ? "bike" : "buggy";
paint();
showMenu(false);

window.addEventListener("load", () => {
  if (!screen.hidden) {
    start.focus();
  }
});

function ensureState() {
  const existing = document.getElementById("silt-menu-state");
  if (existing) {
    return existing;
  }
  const node = document.createElement("div");
  node.id = "silt-menu-state";
  node.hidden = true;
  node.dataset.phase = "menu";
  node.dataset.vehicle = "buggy";
  node.dataset.action = "";
  document.body.append(node);
  return node;
}

function rideButton(id, label, aria) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "ride";
  button.dataset.ride = id;
  button.textContent = label;
  button.setAttribute("aria-label", aria);
  button.addEventListener("click", () => choose(id));
  return button;
}

function actionButton(className, label, action) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = className;
  button.textContent = label;
  button.addEventListener("click", () => send(action));
  return button;
}

function choose(next) {
  ride = next === "bike" ? "bike" : "buggy";
  state.dataset.vehicle = ride;
  paint();
}

function paint() {
  for (const button of [buggy, bike]) {
    const on = button.dataset.ride === ride;
    button.classList.toggle("is-selected", on);
    button.setAttribute("aria-pressed", on ? "true" : "false");
  }
}

function send(action) {
  state.dataset.vehicle = ride;
  state.dataset.action = action;
  if (action === "start") {
    showRace();
  } else {
    showMenu(true);
  }
}

function showMenu(focusStart) {
  state.dataset.phase = "menu";
  screen.hidden = false;
  raceRestart.hidden = true;
  if (focusStart) {
    start.focus();
  }
}

function showRace() {
  state.dataset.phase = "race";
  screen.hidden = true;
  raceRestart.hidden = false;
  canvas?.focus();
}
