import json
import logging
import os
import re
from html.parser import HTMLParser
from typing import Dict, List

logger = logging.getLogger(__name__)


class EventFlagHTMLParser(HTMLParser):
    """Simple HTML parser to extract event flag information from DataCrystal HTML."""

    def __init__(self) -> None:
        super().__init__()
        self.in_table = False
        self.in_row = False
        self.current_cells: List[str] = []
        self.rows: List[List[str]] = []

    def handle_starttag(self, tag: str, attrs):
        if tag == "table":
            for name, value in attrs:
                if name == "class" and "wikitable" in value:
                    self.in_table = True
        elif tag == "tr" and self.in_table:
            self.in_row = True
            self.current_cells = []
        elif tag == "td" and self.in_row:
            self.current_cells.append("")

    def handle_endtag(self, tag: str):
        if tag == "table" and self.in_table:
            self.in_table = False
        elif tag == "tr" and self.in_row:
            if self.current_cells:
                self.rows.append(self.current_cells)
            self.in_row = False
        elif tag == "td" and self.in_row:
            pass

    def handle_data(self, data: str):
        if self.in_row and self.current_cells:
            self.current_cells[-1] += data.strip()


def parse_map_constants(path: str, prefix: str) -> Dict[str, int]:
    """Parse assembly constant definitions starting with a given prefix."""
    constants: Dict[str, int] = {}
    if not os.path.exists(path):
        logger.warning("%s not found. Skipping.", path)
        return constants

    pattern = re.compile(rf"^\s*{prefix}([A-Z0-9_]+)\s+EQU\s+\$?([0-9A-Fa-f]+)")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            m = pattern.match(line)
            if m:
                name = m.group(1).replace("_", " ").title()
                value = int(m.group(2), 16)
                constants[name] = value
    return constants


def parse_items(path: str) -> Dict[str, int]:
    return parse_map_constants(path, "ITEM_")


def parse_maps(path: str) -> Dict[str, int]:
    return parse_map_constants(path, "MAP_")


def parse_event_flags(path: str) -> Dict[str, int]:
    """Parse event flag offsets from a saved DataCrystal file.

    The file may be the HTML table (`ram_map.html`) or a text export
    (`ram_map.txt`). Each entry in the resulting dictionary maps the
    description text (e.g. ``"Event Flag 123: Some Event"``) to the
    hex offset.
    """

    flags: Dict[str, int] = {}
    if not os.path.exists(path):
        logger.warning("%s not found. Event flags not extracted.", path)
        return flags

    if path.endswith(".html"):
        parser = EventFlagHTMLParser()
        with open(path, "r", encoding="utf-8") as f:
            parser.feed(f.read())

        for row in parser.rows:
            if len(row) >= 3 and row[2].startswith("Event Flag"):
                try:
                    offset_str = row[0]
                    description = row[2]
                    offset = int(offset_str.strip("$"), 16)
                    flags[description] = offset
                except ValueError:
                    continue
    else:
        pattern = re.compile(r"^\s*(?:\$|0x)?([0-9A-Fa-f]{4})\s+(Event Flag.*)")
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                m = pattern.match(line)
                if m:
                    offset = int(m.group(1), 16)
                    description = m.group(2).strip()
                    flags[description] = offset
    return flags


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    os.makedirs("data", exist_ok=True)

    map_ids = parse_maps("maps.asm")
    item_ids = parse_items("items.asm")
    # Default to the plain text export from the community disassembly.
    event_flags = parse_event_flags("ram_map.txt")

    with open("data/map_ids.json", "w", encoding="utf-8") as f:
        json.dump(map_ids, f, indent=2)
    with open("data/item_ids.json", "w", encoding="utf-8") as f:
        json.dump(item_ids, f, indent=2)
    with open("data/event_flags.json", "w", encoding="utf-8") as f:
        json.dump(event_flags, f, indent=2)


if __name__ == "__main__":
    main()
