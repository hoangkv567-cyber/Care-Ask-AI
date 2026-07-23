import os
import yaml
from pathlib import Path

# Thư mục gốc dự án (cha của thư mục src/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv():
    """Nạp các biến từ file .env ở gốc dự án vào os.environ (không ghi đè biến đã có).
    Tự cài đặt, không cần thư viện python-dotenv."""
    env_path = _PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                # Không ghi đè biến môi trường đã được set sẵn ở hệ thống
                if key and key not in os.environ:
                    os.environ[key] = value
    except Exception as e:
        print(f"Cảnh báo: không đọc được .env: {e}")


def _apply_env_secrets(config: dict):
    """Đổ các secret từ biến môi trường vào config (ưu tiên env > giá trị trong yaml).
    Nhờ đó secrets KHÔNG cần nằm trong config.yaml (tránh lộ khi commit)."""
    config = config or {}

    neo4j = config.setdefault("neo4j", {})
    if os.getenv("NEO4J_URI"):
        neo4j["uri"] = os.getenv("NEO4J_URI")
    if os.getenv("NEO4J_USER"):
        neo4j["user"] = os.getenv("NEO4J_USER")
    if os.getenv("NEO4J_PASSWORD"):
        neo4j["password"] = os.getenv("NEO4J_PASSWORD")

    if os.getenv("HUGGINGFACE_TOKEN"):
        config.setdefault("huggingface", {})["token"] = os.getenv("HUGGINGFACE_TOKEN")
    if os.getenv("OPENROUTER_API_KEY"):
        config.setdefault("openrouter", {})["api_key"] = os.getenv("OPENROUTER_API_KEY")
    if os.getenv("REQUESTY_API_KEY"):
        config.setdefault("requesty", {})["api_key"] = os.getenv("REQUESTY_API_KEY")
    if os.getenv("REQUESTY_VISION_API_KEY"):
        config.setdefault("requesty", {})["vision_api_key"] = os.getenv("REQUESTY_VISION_API_KEY")
    if os.getenv("DASHSCOPE_API_KEY"):
        config.setdefault("dashscope", {})["api_key"] = os.getenv("DASHSCOPE_API_KEY")

    if os.getenv("TCM_CSV_PATH"):
        config.setdefault("dataset", {})["csv_path"] = os.getenv("TCM_CSV_PATH")

    return config


def load_config(config_path="config/config.yaml"):
    """
    Load file cấu hình YAML, sau đó đổ secrets từ .env / biến môi trường vào.

    Args:
        config_path (str): Đường dẫn đến file cấu hình

    Returns:
        dict: Dictionary chứa cấu hình (đã được bơm secrets từ môi trường)
    """
    _load_dotenv()

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f) or {}
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file cấu hình '{config_path}'")
        print("Sử dụng cấu hình mặc định (secrets lấy từ biến môi trường/.env)...")
        config = {
            "ollama": {"model": "llava:7b"},
            "neo4j": {},
            "mapping": {"symptom_to_syndrome": "data/mapping/symptom_to_syndrome.json"},
            "api": {"host": "0.0.0.0", "port": 8000},
        }
    except Exception as e:
        print(f"Lỗi khi đọc file cấu hình: {e}")
        config = {}

    config = _apply_env_secrets(config)

    # Kiểm tra tối thiểu: cảnh báo nếu thiếu thông tin Neo4j
    if not config.get("neo4j", {}).get("password"):
        print(
            "Cảnh báo: Chưa có NEO4J_PASSWORD. Hãy tạo file .env (xem .env.example) "
            "hoặc set biến môi trường trước khi chạy."
        )

    return config
