// Copy buttons on command blocks.
document.querySelectorAll(".run .copy").forEach((btn) => {
  btn.addEventListener("click", async () => {
    const text = btn.parentElement.querySelector("code").innerText.replace(/^\$ /, "");
    try {
      await navigator.clipboard.writeText(text);
      btn.textContent = "copied";
    } catch (e) {
      btn.textContent = "select + copy";
    }
    setTimeout(() => (btn.textContent = "copy"), 1500);
  });
});
