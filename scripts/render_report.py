"""Render a measurement JSONL file as one self-contained offline HTML report.

Input is the format the runners already write: an optional metadata object
plus measurement rows. Numeric columns get inline bars scaled to the column
maximum; everything else is escaped text. No server, no CDN, no network: open
the generated file in a browser. Raw JSONL stays the source of truth.
"""

import argparse
import html
import json
from pathlib import Path

PREFERRED = ("case", "type", "repeat", "prompt_tokens", "cached_tokens",
             "processed_prompt_tokens", "new_prompt_tokens", "prepare_seconds",
             "first_token_seconds", "first_content_seconds", "decode_seconds",
             "seconds", "generated", "generation_tokens", "commit_published",
             "peak_mlx_bytes", "active_mlx_bytes")


def load_rows(path):
    metadata, rows = {}, []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("type") == "metadata" and not metadata:
            metadata = row
        else:
            rows.append(row)
    return metadata, rows


def is_scalar(value):
    return value is None or isinstance(value, (str, int, float, bool))


def columns(rows):
    present = [key for key in PREFERRED
               if any(key in row and is_scalar(row[key]) for row in rows)]
    extra = sorted({key for row in rows for key, value in row.items()
                    if key not in present and is_scalar(value)})
    return present + extra


def _cell(value, maximum):
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)) and maximum:
        share = min(100.0, abs(value) / maximum * 100.0)
        return (f'<span class="bar" style="width:{share:.1f}%"></span>'
                f'<span class="value">{html.escape(str(value))}</span>')
    return html.escape(str(value))


def render(metadata, rows, title):
    table_columns = columns(rows)
    maxima = {key: max((abs(row[key]) for row in rows
                        if isinstance(row.get(key), (int, float))
                        and not isinstance(row.get(key), bool)), default=0)
              for key in table_columns}
    parts = ["<!doctype html>", "<html><head><meta charset='utf-8'>",
             f"<title>{html.escape(title)}</title>",
             "<style>body{font:14px/1.45 -apple-system,Helvetica,Arial,sans-serif;margin:24px;color:#222}"
             "table{border-collapse:collapse;margin:12px 0;width:100%}"
             "th,td{border:1px solid #ddd;padding:4px 8px;text-align:right;vertical-align:top}"
             "th{background:#f4f4f4;text-align:center}td:first-child,th:first-child{text-align:left}"
             "td{position:relative}.bar{position:absolute;left:0;top:0;bottom:0;background:#dbeafe;z-index:0}"
             ".value{position:relative;z-index:1}h1{font-size:20px}h2{font-size:16px;margin-top:24px}"
             ".meta td{text-align:left;max-width:640px;overflow-wrap:anywhere}</style></head><body>",
             f"<h1>{html.escape(title)}</h1>",
             f"<p>{len(rows)} measurements</p>"]
    if metadata:
        parts.append("<h2>Metadata</h2><table class='meta'>")
        for key, value in metadata.items():
            if is_scalar(value):
                parts.append(f"<tr><th>{html.escape(key)}</th>"
                             f"<td>{html.escape(str(value))}</td></tr>")
        parts.append("</table>")
    if rows:
        parts.append("<h2>Measurements</h2><table><tr>")
        parts.extend(f"<th>{html.escape(key)}</th>" for key in table_columns)
        parts.append("</tr>")
        for row in rows:
            parts.append("<tr>")
            for key in table_columns:
                parts.append(f"<td>{_cell(row.get(key), maxima[key])}</td>")
            parts.append("</tr>")
        parts.append("</table>")
    parts.append("</body></html>")
    return "\n".join(parts) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input")
    parser.add_argument("--output", required=True)
    parser.add_argument("--title")
    args = parser.parse_args()
    source = Path(args.input)
    if not source.is_file():
        parser.error(f"input not found: {source}")
    output = Path(args.output)
    if output.exists():
        parser.error(f"refusing to overwrite {output}")
    metadata, rows = load_rows(source)
    title = args.title or source.stem
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render(metadata, rows, title))
    print(f"wrote {output} ({len(rows)} measurements)")


if __name__ == "__main__":
    main()
