(function () {
  const stage = document.getElementById("stage");
  if (!stage) return;

  window.siltThrottle = 0;
  window.siltBrake = 0;
  window.siltSteer = 0;
  window.siltSteerOn = 0;
  window.siltDriven = 0;

  const hud = document.createElement("div");
  hud.id = "touch";
  hud.innerHTML =
    '<div class="cluster left">' +
    '<div class="cap">РУЛЬ</div>' +
    '<button type="button" class="wheel" aria-label="Руль">' +
    '<div class="wheel-rotor">' +
    '<span class="spoke a"></span>' +
    '<span class="spoke b"></span>' +
    '<span class="spoke c"></span>' +
    '<span class="hub"></span>' +
    "</div></button></div>" +
    '<div class="cluster right"><div class="pedals">' +
    '<button type="button" class="pedal brake" data-pedal="brake" aria-label="Тормоз">' +
    '<span class="cap brake-word">ТОРМОЗ</span><span class="knob"></span></button>' +
    '<button type="button" class="pedal gas" data-pedal="gas" aria-label="Газ">' +
    '<span class="cap">ГАЗ</span><span class="knob"></span></button>' +
    "</div></div>";
  stage.appendChild(hud);

  const wheel = hud.querySelector(".wheel");
  const rotor = hud.querySelector(".wheel-rotor");
  const gas = hud.querySelector("[data-pedal=gas]");
  const brake = hud.querySelector("[data-pedal=brake]");

  let throttle = 0;
  let brakeHeld = 0;
  let steer = 0;
  let steerOn = 0;
  let driven = 0;
  let wheelId = null;
  let originX = 0;
  let spring = 0;

  function publish() {
    window.siltThrottle = throttle;
    window.siltBrake = brakeHeld;
    window.siltSteer = steer;
    window.siltSteerOn = steerOn;
    window.siltDriven = driven;
  }

  function markDriven() {
    driven = 1;
    publish();
  }

  function bindHold(el, set) {
    let id = null;
    const down = (event) => {
      if (id !== null || event.button !== 0) return;
      id = event.pointerId;
      try {
        el.setPointerCapture(event.pointerId);
      } catch (err) {}
      el.classList.add("is-down");
      set(1);
      markDriven();
      event.preventDefault();
      event.stopPropagation();
    };
    const up = (event) => {
      if (id === null || event.pointerId !== id) return;
      id = null;
      el.classList.remove("is-down");
      set(0);
      publish();
    };
    el.addEventListener("pointerdown", down);
    el.addEventListener("pointerup", up);
    el.addEventListener("pointercancel", up);
    el.addEventListener("lostpointercapture", up);
  }

  bindHold(gas, (value) => {
    throttle = value;
  });
  bindHold(brake, (value) => {
    brakeHeld = value;
  });

  const fullLock = 78;

  function paintWheel() {
    rotor.style.transform = "rotate(" + steer * 64 + "deg)";
  }

  function springHome() {
    const from = steer;
    steer = 0;
    steerOn = 0;
    publish();
    const started = performance.now();
    spring += 1;
    const token = spring;
    const step = (now) => {
      if (token !== spring || wheelId !== null) return;
      const t = Math.min(1, (now - started) / 140);
      const eased = 1 - Math.pow(1 - t, 3);
      rotor.style.transform = "rotate(" + from * (1 - eased) * 64 + "deg)";
      if (t < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }

  wheel.addEventListener("pointerdown", (event) => {
    if (event.button !== 0) return;
    try {
      wheel.setPointerCapture(event.pointerId);
    } catch (err) {}
    wheelId = event.pointerId;
    originX = event.clientX - steer * fullLock;
    spring += 1;
    steerOn = 1;
    wheel.classList.add("is-down");
    markDriven();
    event.preventDefault();
    event.stopPropagation();
  });

  wheel.addEventListener("pointermove", (event) => {
    if (event.pointerId !== wheelId) return;
    const next = (event.clientX - originX) / fullLock;
    steer = Math.max(-1, Math.min(1, next));
    paintWheel();
    publish();
  });

  const endWheel = (event) => {
    if (wheelId === null || event.pointerId !== wheelId) return;
    wheelId = null;
    wheel.classList.remove("is-down");
    springHome();
  };
  wheel.addEventListener("pointerup", endWheel);
  wheel.addEventListener("pointercancel", endWheel);
  wheel.addEventListener("lostpointercapture", endWheel);

  hud.addEventListener("contextmenu", (event) => event.preventDefault());
  publish();
})();
