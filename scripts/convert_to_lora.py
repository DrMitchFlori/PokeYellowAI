import json
from pathlib import Path
import argparse

def convert(input_path: Path, output_path: Path) -> None:
    """Convert conversation dataset to simple prompt/response pairs."""
    with input_path.open('r', encoding='utf-8') as fin, output_path.open('w', encoding='utf-8') as fout:
        for line in fin:
            if not line.strip():
                continue
            data = json.loads(line)
            prompt = ''
            response = ''
            for msg in data.get('conversations', []):
                if msg.get('from') == 'human':
                    prompt = msg.get('value', '')
                elif msg.get('from') == 'gpt':
                    response = msg.get('value', '')
            record = {
                'prompt': prompt,
                'response': response
            }
            fout.write(json.dumps(record, ensure_ascii=False) + '\n')


def main() -> None:
    parser = argparse.ArgumentParser(description='Convert dataset to LoRA format')
    parser.add_argument('input', type=Path, help='Input jsonl file')
    parser.add_argument('output', type=Path, help='Output jsonl file')
    args = parser.parse_args()
    convert(args.input, args.output)


if __name__ == '__main__':
    main()
