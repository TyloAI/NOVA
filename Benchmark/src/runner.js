import puppeteer from 'puppeteer';
import { existsSync } from 'fs';
import { platform } from 'os';

function resolveChromePath() {
  const envPath = process.env.PUPPETEER_EXECUTABLE_PATH || process.env.CHROME_PATH;
  if (envPath && existsSync(envPath)) return envPath;

  const candidates = [];
  const plt = platform();
  if (plt === 'darwin') {
    candidates.push(
      '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
      '/Applications/Google Chrome Canary.app/Contents/MacOS/Google Chrome Canary',
    );
  } else if (plt === 'win32') {
    candidates.push(
      'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
      'C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe',
    );
  } else {
    candidates.push('/usr/bin/google-chrome', '/usr/bin/chromium-browser', '/usr/bin/chromium');
  }
  return candidates.find((p) => existsSync(p)) || null;
}

function parseConsolePayload(text) {
  if (text.startsWith('[PROGRESS]')) {
    const [, value, ...rest] = text.split(' ');
    const pct = Number.parseFloat(value);
    const note = rest.join(' ').trim();
    return { type: 'progress', pct: Number.isNaN(pct) ? 0 : pct, note };
  }
  if (text.startsWith('[STATUS]')) {
    return { type: 'status', message: text.replace('[STATUS]', '').trim() };
  }
  if (text.startsWith('[ERROR]')) {
    return { type: 'error', message: text.replace('[ERROR]', '').trim() };
  }
  return { type: 'log', message: text };
}

export async function launchBenchmark({ url, headless = true, onProgress, onStatus, onConsole }) {
  const executablePath = resolveChromePath();
  const launchOptions = {
    headless: headless ? 'new' : false,
    args: [
      '--enable-unsafe-webgpu',
      '--use-gl=angle',
      '--enable-features=Vulkan',
    ],
    defaultViewport: { width: 1400, height: 900 },
  };

  if (executablePath) {
    launchOptions.executablePath = executablePath;
  } else {
    launchOptions.channel = 'chrome'; // fall back to system Chrome channel
  }

  let browser;
  try {
    browser = await puppeteer.launch(launchOptions);
  } catch (err) {
    const hint = executablePath
      ? `Chrome executable failed at ${executablePath}.`
      : 'No bundled Chrome found; set PUPPETEER_EXECUTABLE_PATH or run `npx puppeteer browsers install chrome`.';
    throw new Error(`${err.message} ${hint}`);
  }
  const page = await browser.newPage();

  page.on('console', (message) => {
    const parsed = parseConsolePayload(message.text());
    switch (parsed.type) {
      case 'progress':
        if (onProgress) onProgress(parsed.pct, parsed.note);
        break;
      case 'status':
        if (onStatus) onStatus(parsed.message);
        break;
      case 'error':
        if (onConsole) onConsole(`error: ${parsed.message}`);
        break;
      default:
        if (onConsole) onConsole(parsed.message);
    }
  });

  page.on('pageerror', (err) => {
    if (onConsole) onConsole(`page error: ${err.message || err.toString()}`);
  });

  await page.goto(url, { waitUntil: 'domcontentloaded' });

  const handle = await page.waitForFunction(
    () => {
      const result = globalThis.__BENCHMARK_RESULT__;
      return result && result.done ? result : null;
    },
    { timeout: 0, polling: 1000 },
  );

  const data = await handle.jsonValue();
  await browser.close();
  return data;
}
