const menuButton = document.querySelector("[data-menu-button]");
const mobileMenu = document.querySelector("[data-mobile-menu]");

function setMenu(open) {
  if (!menuButton || !mobileMenu) return;
  menuButton.setAttribute("aria-expanded", String(open));
  mobileMenu.hidden = !open;
}

menuButton?.addEventListener("click", () => {
  setMenu(menuButton.getAttribute("aria-expanded") !== "true");
});

mobileMenu?.addEventListener("click", (event) => {
  if (event.target.closest("a")) setMenu(false);
});

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") setMenu(false);
});

async function copyText(button) {
  let text = button.dataset.copyText;
  if (button.dataset.copyCode !== undefined) {
    text = button.closest(".code-block")?.querySelector("code")?.innerText;
  }
  if (button.dataset.copyUrl) {
    const response = await fetch(button.dataset.copyUrl);
    if (!response.ok) throw new Error("source unavailable");
    text = await response.text();
  }
  if (!text) throw new Error("nothing to copy");
  let copied = false;
  if (navigator.clipboard?.writeText) {
    try {
      await navigator.clipboard.writeText(text);
      copied = true;
    } catch {
      copied = false;
    }
  }
  if (!copied) {
    const field = document.createElement("textarea");
    field.value = text;
    field.setAttribute("readonly", "");
    field.className = "copy-fallback";
    document.body.append(field);
    field.select();
    try {
      if (!document.execCommand("copy")) throw new Error("copy unavailable");
    } finally {
      field.remove();
    }
  }
}

document.querySelectorAll("[data-copy]").forEach((button) => {
  const original = button.querySelector("[data-copy-label]")?.textContent || "Copy";
  button.addEventListener("click", async () => {
    const label = button.querySelector("[data-copy-label]");
    try {
      await copyText(button);
      if (label) label.textContent = "Copied";
    } catch {
      if (label) label.textContent = "Select and copy";
    }
    window.setTimeout(() => {
      if (label) label.textContent = original;
    }, 2400);
  });
});

document.querySelectorAll("[data-rail-controls]").forEach((controls) => {
  const rail = document.getElementById(controls.dataset.railControls);
  if (!rail) return;
  controls.querySelectorAll("button").forEach((button) => {
    button.addEventListener("click", () => {
      const direction = button.dataset.direction === "previous" ? -1 : 1;
      rail.scrollBy({ left: direction * Math.min(rail.clientWidth * 0.82, 520), behavior: "smooth" });
    });
  });
  rail.addEventListener("keydown", (event) => {
    if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
    event.preventDefault();
    const direction = event.key === "ArrowLeft" ? -1 : 1;
    rail.scrollBy({ left: direction * 340, behavior: "smooth" });
  });
});
