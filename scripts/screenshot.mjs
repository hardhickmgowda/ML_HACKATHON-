// Headless-Chrome screenshots for the report (no extra dependencies; Node >= 22 for the global WebSocket).
//   node scripts/screenshot.mjs <url> <out.png> [width] [height] [clickTripId|-] [lat,lon,zoom]
// With clickTripId the PathSnap UI is driven like a user: wait for the trip list, pick the trip, press "Match route".
import { spawn } from "node:child_process";
import { writeFileSync, mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const [url, out, w = "1400", h = "850", tripArg, view] = process.argv.slice(2);
const trip = tripArg && tripArg !== "-" ? tripArg : null;
const CHROME = process.env.CHROME || "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const port = 9300 + Math.floor(Math.random() * 500);
const chrome = spawn(CHROME, ["--headless=new", `--remote-debugging-port=${port}`, `--window-size=${w},${h}`,
  "--hide-scrollbars", `--user-data-dir=${mkdtempSync(join(tmpdir(), "cdp-"))}`, "about:blank"], { stdio: "ignore" });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let target;
for (let i = 0; i < 50 && !target; i++) {
  await sleep(200);
  try { target = (await (await fetch(`http://127.0.0.1:${port}/json`)).json()).find((t) => t.type === "page"); } catch {}
}
const ws = new WebSocket(target.webSocketDebuggerUrl);
await new Promise((r) => (ws.onopen = r));
let id = 0; const pending = new Map();
ws.onmessage = (e) => { const m = JSON.parse(e.data); if (m.id && pending.has(m.id)) { pending.get(m.id)(m.result); pending.delete(m.id); } };
const send = (method, params = {}) => new Promise((r) => { const i = ++id; pending.set(i, r); ws.send(JSON.stringify({ id: i, method, params })); });
const evalJs = async (expr) => (await send("Runtime.evaluate", { expression: expr, awaitPromise: true, returnByValue: true }))?.result?.value;

await send("Emulation.setDeviceMetricsOverride", { width: +w, height: +h, deviceScaleFactor: 2, mobile: false });
await send("Page.enable");
await send("Page.navigate", { url });
await sleep(2500);
if (trip) {
  for (let i = 0; i < 40 && !(await evalJs(`document.querySelectorAll('#trip option').length`)); i++) await sleep(250);
  await evalJs(`document.getElementById('trip').value='${trip}'; document.getElementById('l_clf').click(); document.getElementById('run').click()`);
  for (let i = 0; i < 80 && !(await evalJs(`/Done in/.test(document.getElementById('status').textContent)`)); i++) await sleep(250);
}
if (view) { // zoom a folium map: its Leaflet map object is a global named map_<hash>
  const [la, lo, z] = view.split(",").map(Number);
  await evalJs(`Object.keys(window).filter(k => k.startsWith('map_')).forEach(k => window[k].setView([${la}, ${lo}], ${z}))`);
}
await sleep(3500); // map tiles
const shot = await send("Page.captureScreenshot", { format: "png" });
writeFileSync(out, Buffer.from(shot.data, "base64"));
ws.close(); chrome.kill();
console.log("saved", out);
