# main.py
import sys
import json
import argparse
from src.qa_system import TCMQA
from src.config_loader import load_config

def main():
    # Reconfigure stdout to use utf-8 to handle Unicode characters on Windows terminal
    if sys.stdout.encoding != 'utf-8':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except AttributeError:
            import io
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    parser = argparse.ArgumentParser(description="Hệ thống hỏi đáp Đông y dùng Qwen 2.5")
    parser.add_argument("question", help="Câu hỏi của bạn")
    parser.add_argument("--config", default="config/config.yaml", help="File configuration")
    parser.add_argument("--output", help="File output JSON")
    args = parser.parse_args()

    config = load_config(args.config)
    qa = TCMQA(config=config)
    
    result = qa.ask(args.question)
    
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"Đã lưu kết quả vào {args.output}")
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    
    qa.close()

if __name__ == "__main__":
    main()
