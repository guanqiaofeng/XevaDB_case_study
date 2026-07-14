import fs from "node:fs/promises";
import path from "node:path";
process.on("uncaughtException", (error) => {
  console.error("UNCAUGHT", error?.name, error?.message);
  process.exit(1);
});
process.on("unhandledRejection", (error) => {
  console.error("UNHANDLED", error?.name, error?.message);
  process.exit(1);
});

const { FileBlob, SpreadsheetFile } = await import("@oai/artifact-tool");

const inputPath = "/Users/guanqiaofeng/Library/Containers/com.microsoft.Outlook/Data/tmp/Outlook Temp/mmc3 (1).xlsx";
const renderDir = "/tmp/mmc3_renders";

const input = await FileBlob.load(inputPath);
const workbook = await SpreadsheetFile.importXlsx(input);

const sheetSummary = await workbook.inspect({
  kind: "sheet",
  include: "id,name",
  maxChars: 12000,
});
console.log("SHEETS");
console.log(sheetSummary.ndjson);

const overview = await workbook.inspect({
  kind: "workbook,sheet,table",
  maxChars: 30000,
  tableMaxRows: 12,
  tableMaxCols: 60,
  tableMaxCellChars: 120,
});
console.log("OVERVIEW");
console.log(overview.ndjson);

await fs.mkdir(renderDir, { recursive: true });
for (let i = 0; i < workbook.worksheets.items.length; i += 1) {
  const sheet = workbook.worksheets.getItemAt(i);
  const used = sheet.getUsedRange(true);
  const usedAddress = used?.address ?? "A1";
  console.log(`USED_RANGE\t${sheet.name}\t${usedAddress}`);
  const rendered = await workbook.render({
    sheetName: sheet.name,
    range: usedAddress,
    scale: 1,
    format: "png",
  });
  const bytes = new Uint8Array(await rendered.arrayBuffer());
  const safeName = sheet.name.replace(/[^A-Za-z0-9._-]+/g, "_");
  await fs.writeFile(path.join(renderDir, `${i + 1}_${safeName}.png`), bytes);
}
