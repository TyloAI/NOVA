#!/usr/bin/env node
import { Command } from 'commander';
import inquirer from 'inquirer';
import chalk from 'chalk';
import { dirname, join, resolve } from 'path';
import { fileURLToPath } from 'url';
import { existsSync, mkdirSync } from 'fs';
import { startServer } from '../src/server.js';
import { launchBenchmark } from '../src/runner.js';
import { generateReport } from '../src/report.js';

const __dirname = dirname(fileURLToPath(import.meta.url));

const program = new Command()
  .name('benchmark')
  .description('N.O.V.A. Protocol browser-edge benchmark (CLI)')
  .option('-s, --seed <number>', 'Seed for deterministic runs', (value) => parseInt(value, 10), Date.now())
  .option('-m, --models <path>', 'Path to local models directory exposed to the browser', join(process.cwd(), 'models'))
  .option('--headful', 'Run Chrome with UI (debug)', false)
  .option('--no-open-report', 'Skip auto-opening the generated HTML report')
  .option('--keep-server', 'Keep the HTTP server alive after the run', false)
  .version('0.1.0');

program.parse(process.argv);
const options = program.opts();

async function main() {
  console.clear();
  console.log(chalk.cyan(`
███╗   ██╗ ██████╗ ██╗   ██╗ █████╗ 
████╗  ██║██╔═══██╗██║   ██║██╔══██╗
██╔██╗ ██║██║   ██║██║   ██║███████║
██║╚██╗██║██║   ██║██║   ██║██╔══██║
██║ ╚████║╚██████╔╝╚██████╔╝██║  ██║
╚═╝  ╚═══╝ ╚═════╝  ╚═════╝ ╚═╝  ╚═╝
`));
  console.log(chalk.magenta('N.O.V.A. Protocol — Browser Edge Benchmark\n'));

  const seed = Number.isNaN(options.seed) ? Date.now() : Number(options.seed);
  const modelsDir = resolve(options.models);

  const modelChoices = ['N.O.V.A. (builtin)', 'Transformer (baseline)', 'Custom model (local)'];
  const answers = await inquirer.prompt([
    {
      type: 'list',
      name: 'modelA',
      message: 'Select Model A (Arena A)',
      choices: ['N.O.V.A. (builtin)', 'Custom model (local)'],
      default: 'N.O.V.A. (builtin)',
    },
    {
      type: 'input',
      name: 'modelAPath',
      message: 'Entry script for Model A (served from /models)',
      when: (res) => res.modelA === 'Custom model (local)',
    },
    {
      type: 'list',
      name: 'modelB',
      message: 'Select Model B (Arena B)',
      choices: modelChoices,
      default: 'Transformer (baseline)',
    },
    {
      type: 'input',
      name: 'modelBPath',
      message: 'Entry script for Model B (served from /models)',
      when: (res) => res.modelB === 'Custom model (local)',
    },
    {
      type: 'list',
      name: 'reportStyle',
      message: 'Report output',
      choices: ['Full (HTML)', 'Minimal (JSON only)'],
      default: 'Full (HTML)',
    },
  ]);

  const { port, close, baseUrl } = await startServer({
    staticDir: join(__dirname, '../src/frontend'),
    modelsDir,
  });

  console.log(chalk.cyan(`\n[server] Listening on ${baseUrl} with COOP/COEP headers`));
  const params = new URLSearchParams({
    seed: String(seed),
    modelA: answers.modelA,
    modelB: answers.modelB,
    modelAPath: answers.modelAPath || '',
    modelBPath: answers.modelBPath || '',
  });
  const targetUrl = `${baseUrl}/?${params.toString()}`;

  const progress = (pct, message) => {
    const pretty = pct.toFixed(0).padStart(3, ' ');
    const suffix = message ? ` ${message}` : '';
    process.stdout.write(`\r[arena] ${pretty}%${suffix}   `);
  };

  const data = await launchBenchmark({
    url: targetUrl,
    headless: !options.headful,
    onProgress: (pct, note) => progress(pct, note),
    onStatus: (msg) => console.log(`\n[log] ${msg}`),
    onConsole: (msg) => console.log(`\n[console] ${msg}`),
  });

  process.stdout.write('\n');
  console.log(chalk.green('[arena] Benchmark finished'));

  let reportPath = null;
  if (answers.reportStyle === 'Full (HTML)') {
    const outDir = resolve(process.cwd());
    if (!existsSync(outDir)) {
      mkdirSync(outDir, { recursive: true });
    }
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    reportPath = join(outDir, `benchmark-report-${timestamp}.html`);
    await generateReport({
      data,
      templatePath: join(__dirname, '../templates/template.html'),
      outputPath: reportPath,
      openReport: options.openReport,
    });
    console.log(chalk.yellow(`[report] Wrote ${reportPath}`));
  } else {
    console.log(JSON.stringify(data, null, 2));
  }

  if (!options.keepServer) {
    await close();
  } else {
    console.log(chalk.gray(`[server] Keeping server alive on port ${port}`));
  }
  return reportPath;
}

main().catch((err) => {
  console.error(chalk.red('\n[error]'), err);
  process.exitCode = 1;
});
