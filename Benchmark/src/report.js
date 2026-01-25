import { readFile, writeFile } from 'fs/promises';
import open from 'open';

export async function generateReport({ data, templatePath, outputPath, openReport = true }) {
  const template = await readFile(templatePath, 'utf-8');
  const serialized = JSON.stringify(data, null, 2).replace(/</g, '\\u003c');
  const html = template.replace('__REPORT_DATA__', serialized);
  await writeFile(outputPath, html, 'utf-8');

  if (openReport) {
    await open(outputPath);
  }
  return outputPath;
}
