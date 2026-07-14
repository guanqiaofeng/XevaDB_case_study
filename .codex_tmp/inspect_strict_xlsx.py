from __future__ import annotations

import json
import gzip
import re
import zipfile
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree as ET


INPUT = Path(
    "/Users/guanqiaofeng/Library/Containers/com.microsoft.Outlook/Data/tmp/Outlook Temp/mmc3 (1).xlsx"
)
NS = {"m": "http://purl.oclc.org/ooxml/spreadsheetml/main"}


def column_index(cell_reference: str) -> int:
    letters = re.match(r"[A-Z]+", cell_reference).group(0)
    value = 0
    for letter in letters:
        value = value * 26 + ord(letter) - 64
    return value - 1


with zipfile.ZipFile(INPUT) as archive:
    shared_root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    shared_strings = [
        "".join(node.text or "" for node in item.findall(".//m:t", NS))
        for item in shared_root.findall("m:si", NS)
    ]

    sheet_root = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
    parsed_rows: list[list[object | None]] = []
    max_columns = 0
    for row in sheet_root.findall(".//m:sheetData/m:row", NS):
        values: dict[int, object | None] = {}
        for cell in row.findall("m:c", NS):
            idx = column_index(cell.attrib["r"])
            cell_type = cell.attrib.get("t")
            value_node = cell.find("m:v", NS)
            raw = None if value_node is None else value_node.text
            if raw is None:
                value: object | None = None
            elif cell_type == "s":
                value = shared_strings[int(raw)]
            elif cell_type == "b":
                value = raw == "1"
            elif cell_type in {"str", "inlineStr"}:
                value = raw
            else:
                try:
                    number = float(raw)
                    value = int(number) if number.is_integer() else number
                except ValueError:
                    value = raw
            values[idx] = value
            max_columns = max(max_columns, idx + 1)
        parsed_rows.append([values.get(i) for i in range(max_columns)])

for row in parsed_rows:
    row.extend([None] * (max_columns - len(row)))

print("DIMENSIONS", len(parsed_rows), max_columns)
print("FIRST_ROWS")
for index, row in enumerate(parsed_rows[:8], start=1):
    print(index, json.dumps(row, ensure_ascii=False))

header_index = next(
    index
    for index, row in enumerate(parsed_rows)
    if any(
        str(value).strip().lower()
        in {"resid", "ispy2 id", "patient_id", "patient identifier"}
        for value in row
        if value is not None
    )
)
headers = [str(value).strip() if value is not None else f"unnamed_{i + 1}" for i, value in enumerate(parsed_rows[header_index])]
data = parsed_rows[header_index + 1 :]
print("HEADER_ROW", header_index + 1)
print("HEADERS", json.dumps(headers, ensure_ascii=False))
print("DATA_ROWS", len(data))

keywords = ("pcr", "drfs", "surv", "event", "time", "arm", "her2", "hr", "receptor", "mp", "stage", "age", "race")
for column_index_value, header in enumerate(headers):
    if any(keyword in header.lower() for keyword in keywords):
        values = [
            row[column_index_value]
            for row in data
            if row[column_index_value] not in (None, "", "NA")
        ]
        counts = Counter(str(value) for value in values)
        print(
            "COLUMN",
            json.dumps(
                {
                    "index": column_index_value + 1,
                    "header": header,
                    "nonmissing": len(values),
                    "missing": len(data) - len(values),
                    "unique": len(counts),
                    "top_values": counts.most_common(12),
                },
                ensure_ascii=False,
            ),
        )

identifier_index = headers.index("Patient Identifier")
identifiers = [str(row[identifier_index]) for row in data if row[identifier_index] not in (None, "", "NA")]
print("IDENTIFIERS", json.dumps({
    "nonmissing": len(identifiers),
    "unique": len(set(identifiers)),
    "duplicates": len(identifiers) - len(set(identifiers)),
    "examples": identifiers[:10],
}))

expression_path = Path("/tmp/GSE194040_geneLevel.txt.gz")
if expression_path.exists():
    with gzip.open(expression_path, "rt") as expression_file:
        expression_identifiers = expression_file.readline().rstrip("\n").split("\t")
    clinical_set = set(identifiers)
    expression_set = set(expression_identifiers)
    print("EXPRESSION_JOIN", json.dumps({
        "expression_columns": len(expression_identifiers),
        "expression_unique_ids": len(expression_set),
        "matched_clinical_ids": len(clinical_set & expression_set),
        "clinical_ids_without_expression": sorted(clinical_set - expression_set),
        "expression_ids_without_clinical": sorted(expression_set - clinical_set),
    }))
