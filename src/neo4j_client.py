import os
from neo4j import GraphDatabase
import logging

logger = logging.getLogger(__name__)

class Neo4jTCMClient:
    def __init__(self, uri=None, user=None, password=None):
        # Ưu tiên tham số truyền vào; nếu không có thì lấy từ biến môi trường (.env)
        uri = uri or os.getenv("NEO4J_URI")
        user = user or os.getenv("NEO4J_USER")
        password = password or os.getenv("NEO4J_PASSWORD")
        if not (uri and user and password):
            raise ValueError(
                "Thiếu thông tin kết nối Neo4j. Hãy đặt NEO4J_URI/NEO4J_USER/NEO4J_PASSWORD "
                "trong file .env (xem .env.example) hoặc truyền trực tiếp."
            )
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        logger.info(f"Kết nối Neo4j thành công: {uri}")

    def close(self):
        self.driver.close()
        logger.info("Đã đóng kết nối Neo4j")

    def get_treatment_by_syndrome(self, syndrome_name: str, disease_name: str = None) -> dict:
        """Lấy bài thuốc và vị thuốc theo hội chứng (và bệnh lý nếu có)"""
        with self.driver.session() as session:
            # [FIX CASE-INSENSITIVE] KG có nhiều nhóm HoiChung trùng tên khác hoa/thường —
            # khớp toLower để không trượt bài thuốc; p.hoi_chung = s.name giữ per-node.
            # ORDER BY xác định (ưu tiên node khớp ĐÚNG hoa/thường, rồi theo tên bài) trước LIMIT 1
            # để khi có nhiều node biến thể hoa/thường, kết quả KHÔNG phụ thuộc thứ tự Neo4j trả về.
            if disease_name:
                result = session.run(
                    """
                    MATCH (s:HoiChung)-[:ĐƯỢC_ĐIỀU_TRỊ_BẰNG]->(p:BaiThuoc {benh_ly: $disease_name})
                    WHERE toLower(s.name) = toLower($name) AND p.hoi_chung = s.name
                    OPTIONAL MATCH (p)-[:BAO_GỒM]->(v:ViThuoc)
                    WITH s, p, COLLECT(DISTINCT v.name) AS vi_thuoc
                    RETURN s.name AS hoi_chung, p.name AS bai_thuoc, vi_thuoc
                    ORDER BY (s.name = $name) DESC, p.name
                    LIMIT 1
                    """,
                    name=syndrome_name,
                    disease_name=disease_name
                )
            else:
                result = session.run(
                    """
                    MATCH (s:HoiChung)-[:ĐƯỢC_ĐIỀU_TRỊ_BẰNG]->(p:BaiThuoc)
                    WHERE toLower(s.name) = toLower($name)
                    OPTIONAL MATCH (p)-[:BAO_GỒM]->(v:ViThuoc)
                    WITH s, p, COLLECT(DISTINCT v.name) AS vi_thuoc
                    RETURN s.name AS hoi_chung, p.name AS bai_thuoc, vi_thuoc
                    ORDER BY (s.name = $name) DESC, p.name
                    LIMIT 1
                    """,
                    name=syndrome_name
                )
            record = result.single()
            if record:
                return {
                    "hoi_chung": record["hoi_chung"],
                    "bai_thuoc": record["bai_thuoc"],
                    "vi_thuoc": record["vi_thuoc"]
                }
            return None


    def get_all_syndromes(self) -> list:
        """Lấy danh sách tất cả hội chứng"""
        with self.driver.session() as session:
            result = session.run(
                "MATCH (s:HoiChung) RETURN s.name AS name ORDER BY name"
            )
            return [record["name"] for record in result]

    def get_all_symptoms(self) -> list:
        """Lấy danh sách tất cả triệu chứng"""
        with self.driver.session() as session:
            result = session.run(
                "MATCH (t:TrieuChung) RETURN t.name AS name ORDER BY name"
            )
            return [record["name"] for record in result]

    def get_symptoms_by_syndrome(self, syndrome_name: str) -> list:
        """Lấy danh sách triệu chứng theo hội chứng"""
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (s:HoiChung)-[:CÓ_BIỂU_HIỆN]->(t:TrieuChung)
                WHERE toLower(s.name) = toLower($name)
                RETURN t.name AS trieu_chung
                """,
                name=syndrome_name
            )
            return [record["trieu_chung"] for record in result]

    def get_diseases_by_syndrome(self, syndrome_name: str) -> list:
        """Lấy danh sách bệnh có hội chứng này"""
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (b:BenhLy)-[:CHIA_THÀNH]->(s:HoiChung)
                WHERE toLower(s.name) = toLower($name)
                RETURN b.name AS benh_ly
                """,
                name=syndrome_name
            )
            return [record["benh_ly"] for record in result]