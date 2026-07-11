#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/build_disease_gates.py — Sinh data/disease_gates.json từ các luật cổng an toàn bệnh danh.

Chuyển toàn bộ tri thức "cổng chủ chứng bệnh danh" (trước đây hard-code ~460 dòng if/else trong
_validate_disease_safety) thành DỮ LIỆU. Mỗi luật:
  names        : nếu tên bệnh CHỨA bất kỳ chuỗi nào -> áp luật (khớp substring, thường)
  names_regex  : (tùy) khớp tên theo RANH GIỚI TỪ (vd 'ái' không dính 'khái'/'đái')
  requires     : lời khai phải chứa >=1 từ khóa này, nếu không -> LOẠI bệnh. RỖNG = luôn loại
                 (bệnh Tây y thuần: khớp tên là loại, vd Addison/Parkinson).
  kw_wb        : true = khớp requires theo ranh giới từ (\\b); false = substring (mặc định).

Ngữ nghĩa GIỮ NGUYÊN 100% so với bản hard-code (mỗi cổng độc lập, bệnh phải qua HẾT). Thứ tự luật
không ảnh hưởng kết quả boolean. Kiểm chứng bằng scripts/test_gate_equivalence.py.

Chạy:  python scripts/build_disease_gates.py   -> ghi data/disease_gates.json
"""
import os
import sys
import json

OUT = "data/disease_gates.json"


def build():
    gates = []

    # ── Nhóm 0: cổng chủ chứng đặc biệt (tên bệnh = triệu chứng chủ đạo) — substring/substring ──
    gates += [
        {"names": ["xuất hãn", "tự hãn", "đạo hãn", "mồ hôi", "hãn chứng"],
         "requires": ["mồ hôi", "hãn", "đổ mồ hôi", "ra mồ hôi", "vã mồ hôi", "ướt đẫm"]},
        {"names": ["cuồng", "điên", "thao cuồng"],
         "requires": ["kích động", "la hét", "hưng phấn", "nói nhảm", "hoang tưởng",
                      "đập phá", "cuồng", "điên", "mất ngủ", "loạn thần", "rối loạn tâm thần"]},
        {"names": ["ách nghịch", "nấc"], "requires": ["nấc", "ách nghịch"]},
        {"names": ["động kinh", "giản chứng", "kinh phong", "phong giật", "co giật"],
         "requires": ["co giật", "động kinh", "giật", "kinh phong", "sùi bọt mép",
                      "ngã lăn", "mất ý thức", "hôn mê", "cứng người"]},
    ]

    # ── Nhóm 0b: _DEFINING_SYMPTOM_RULES — substring tên, RANH GIỚI TỪ cho requires (kw_wb=True) ──
    defining = [
        (("bất mị", "thất miên", "mất ngủ"),
         ("mất ngủ", "khó ngủ", "không ngủ", "ngủ không", "thất miên", "khó vào giấc", "dễ tỉnh",
          "trằn trọc", "ngủ kém", "ngủ chập chờn", "chập chờn", "tỉnh giấc", "thức giấc", "mộng nhiều",
          "mơ nhiều", "hay mơ", "ngủ hay mơ", "khó đi vào giấc", "khó vào giấc ngủ")),
        (("đầu thống",), ("đau đầu", "nhức đầu", "đầu thống", "đau nửa đầu")),
        (("huyễn vựng",), ("chóng mặt", "hoa mắt", "choáng váng", "váng đầu", "huyễn vựng", "xây xẩm")),
        (("phúc thống",), ("đau bụng", "bụng đau", "phúc thống", "đau quặn bụng", "đau vùng bụng", "bụng dưới đau")),
        (("vị quản thống",),
         ("đau thượng vị", "đau vùng thượng vị", "đau dạ dày", "đau bao tử", "đau bụng", "vị quản thống", "bụng đau")),
        (("tiết tả", "tiêu chảy"), ("tiêu chảy", "đại tiện lỏng", "phân lỏng", "phân nát", "ỉa chảy", "đi lỏng", "tiết tả")),
        (("ẩu thổ", "nôn mửa"), ("nôn", "buồn nôn", "ói", "ẩu thổ")),
        (("táo bón", "tiện bí"), ("táo bón", "đại tiện táo", "khó đại tiện", "phân khô", "tiện bí")),
        (("hiếp thống", "hiếp gian"),
         ("đau sườn", "đau hạ sườn", "đau mạng sườn", "tức sườn", "đau hông sườn", "sườn trướng", "đau liên sườn", "hiếp thống")),
        (("yêu thống",), ("đau lưng", "mỏi lưng", "đau thắt lưng", "yêu thống")),
        (("khái thấu", "khái suyễn"), ("ho", "khái thấu")),
        # Bách nhật khái (ho gà — 百日咳) là bệnh nhiễm có CƠN HO RŨ RƯỢI + TIẾNG RÍT (whoop) kéo dài,
        # KHÔNG được gate chỉ bằng 'ho' như khái thấu thường — nếu không, thể Phong hàn giai đoạn đầu
        # (chảy nước mũi + ho liên tục + rêu trắng mỏng, giống cảm mạo) sẽ dán 'ho gà' cho mọi ca ho
        # khan. Đòi hỏi dấu ho-cơn/tiếng-rít đặc hiệu. Cả 3 thể CSV đều có (ho liên tục/ho cơn/tiếng
        # rít) nên bảo toàn self-recall.
        (("bách nhật khái", "ho gà"),
         ("ho cơn", "cơn ho", "ho từng cơn", "ho thành cơn", "ho rũ rượi", "ho liên tục",
          "ho dồn dập", "ho sặc sụa", "tiếng rít", "rít khi ho", "ho gà", "bách nhật khái",
          "ho kéo dài", "ho dai dẳng", "nôn sau ho", "nôn ra đờm")),
        (("khái huyết",),
         ("máu", "khái huyết", "ho ra máu", "khạc ra máu", "khạc máu", "đờm máu", "đờm có máu",
          "đờm lẫn máu", "ho khạc máu", "máu tươi", "huyết ra")),
        # Khẩu nhãn oa tà (liệt mặt ngoại biên) ĐỊNH NGHĨA bằng dấu LIỆT MẶT: mặt/miệng méo lệch,
        # mắt nhắm không kín, liệt (nửa) mặt. Thể 'Phong hàn' của nó (sợ lạnh, tắc mũi, gáy căng)
        # trùng triệu chứng cảm mạo -> từng dán 'liệt mặt' + kê bài Toàn yết/Cương tàm/Địa long cho
        # ca cảm cúm thường. Cả 6 thể CSV đều có >=1 dấu {méo, liệt mặt/nửa mặt, mắt nhắm không kín}.
        (("khẩu nhãn oa tà", "liệt mặt", "liệt thần kinh mặt", "liệt dây thần kinh"),
         ("méo", "méo miệng", "miệng méo", "mặt méo", "méo lệch", "liệt mặt", "liệt nửa mặt",
          "mắt nhắm không kín", "mắt không nhắm", "không nhắm được mắt", "nhắm không kín",
          "mặt lệch", "nhân trung lệch", "khẩu nhãn oa tà")),
        # Anh lựu (bướu cổ/tuyến giáp) đòi khối/bướu vùng cổ thật; Viêm sai (quai bị) đòi sưng đau
        # vùng mang tai/má — cả hai từng leo top bệnh danh ca hô hấp chỉ nhờ triệu chứng toàn thân
        # chung (sốt, mệt, rêu lưỡi). Mọi thể CSV đều mở đầu bằng dấu định nghĩa nên giữ self-recall.
        (("anh lựu", "bướu cổ", "khí anh", "nhục anh"),
         ("tuyến giáp", "bướu", "bướu cổ", "khối u", "u vùng cổ", "cổ to", "sưng cổ", "cổ sưng",
          "cục ở cổ", "khối ở cổ", "anh lựu")),
        (("viêm sai", "quai bị"),
         ("mang tai", "má sưng", "sưng má", "vùng má", "quai bị", "khó há miệng", "quai hàm",
          "tuyến nước bọt", "viêm sai")),
        # Đởm kết thạch (sỏi đường mật/túi mật) ĐỊNH NGHĨA bằng dấu GAN-MẬT: đau vùng hạ sườn/hông
        # phải, vàng da/mắt, miệng đắng, sỏi mật, đau xoắn quặn. TUYỆT ĐỐI không để khớp chỉ vì
        # 'sốt'+'khô họng' (trùng 'miệng đắng khô họng, sốt sợ lạnh' của các thể) — nếu không, ca
        # cảm cúm hô hấp (ngạt mũi/sổ mũi/ho/sốt) bị dán 'sỏi mật'. Cả 9 thể CSV đều có >=1 dấu
        # {đau sườn/hạ sườn/hông phải, vàng da, miệng đắng} nên requires bảo toàn self-recall.
        (("đởm kết thạch", "sỏi mật", "sỏi đường mật", "sỏi túi mật"),
         ("đau sườn", "hạ sườn", "sườn phải", "hông phải", "đau hông", "hông đau", "mạng sườn",
          "vùng gan", "vàng da", "da vàng", "vàng mắt", "mắt vàng", "hoàng đản", "miệng đắng",
          "sỏi mật", "sỏi đường mật", "sỏi túi mật", "đau xoắn", "quặn", "túi mật")),
        # Hầu ngứa là bệnh danh ĐẶT THEO CHỦ CHỨNG ngứa họng — không được dán chỉ vì bệnh nhân ho
        # (thể Phong hàn 'Họng ngứa, ho lâu, ho khan' từng khớp ca 'ho + sợ gió' không hề ngứa họng
        # rồi chiếm cả Mục 1 lẫn bài thuốc Mục 5). Đòi hỏi lời khai có ngứa họng/hầu thật. Cả 3 thể
        # CSV đều có ('Trong họng ngứa'/'Yết hầu đau và ngứa, khi ngứa thì ho'/'Họng ngứa') nên
        # requires phủ đủ, bảo toàn self-recall.
        (("hầu ngứa",),
         ("ngứa họng", "họng ngứa", "ngứa cổ họng", "cổ họng ngứa", "ngứa trong họng", "ngứa cổ",
          "ngứa hầu", "hầu ngứa", "yết hầu đau và ngứa", "ngứa thì ho", "ngứa rát họng")),
        (("hư lao",),
         ("mệt mỏi", "suy nhược", "gầy sút", "vô lực", "đuối sức", "uể oải", "sụt cân", "người yếu", "hư lao", "bệnh lâu ngày")),
        (("cổ trướng",), ("bụng to", "bụng căng", "bụng trướng to", "báng bụng", "cổ trướng")),
        (("xơ gan", "viêm gan", "gan nhiễm mỡ"),
         ("vùng gan", "gan to", "men gan", "viêm gan", "xơ gan", "vàng da", "da vàng", "mắt vàng",
          "vàng mắt", "cổ trướng", "bụng to", "bụng căng", "đau sườn", "hạ sườn", "tức sườn",
          "sườn đau", "nôn ra máu", "chảy máu cam")),
        (("huyết áp thấp", "huyết áp cao", "cao huyết áp", "tăng huyết áp"),
         ("huyết áp", "tụt huyết áp", "hạ áp", "tụt áp", "tăng áp")),
        (("lao phổi", "phế lao"), ("ho", "ho ra máu", "sốt về chiều", "gầy sút", "lao phổi", "phế lao")),
        (("cao chỉ huyết", "cao huyết chỉ", "chỉ huyết cao"), ("cholesterol", "mỡ máu", "máu nhiễm mỡ", "lipid")),
        (("uất chứng",), ("căng thẳng", "uất ức", "buồn phiền", "trầm cảm", "lo âu", "stress", "tinh thần không", "hay thở dài", "cáu gắt", "tức giận")),
        (("tào tạp", "thôn toan"), ("cồn cào", "ợ chua", "nóng rát", "thôn toan", "tào tạp")),
        (("sán khí", "hàn sán", "hồ sán", "khí sán", "thoát vị"),
         ("bẹn", "bìu", "tinh hoàn", "thoát vị", "sa ruột", "khối phồng", "sán khí",
          "đau bụng dưới", "bụng dưới đau")),
        # Viêm tắc động mạch (thoát thư) định nghĩa bằng THIẾU MÁU CHI THỰC THỂ: đau chi/đau cách hồi
        # (đi đau nghỉ đỡ), chuột rút, tím/hoại tử đầu chi, vết loét không lành. TUYỆT ĐỐI không dùng
        # 'tay chân lạnh'/'chi lạnh' trần trong requires — đó là dấu DƯƠNG HƯ phổ biến nhất, sẽ dán
        # bệnh mạch nặng (Buerger) cho mọi ca hư hàn. Thể 'Khí huyết hư'/'Dương hư ứ trệ' của bệnh này
        # vẫn tự khớp qua dấu đặc hiệu (vết loét, đi đau nghỉ đỡ, chuột rút).
        (("động mạch viêm tắc", "viêm tắc động mạch", "thoát thư", "tắc động mạch"),
         ("đau chi", "đau chân", "đau tay", "đau cách hồi", "đi đau nghỉ đỡ", "đi đau", "chuột rút",
          "khập khiễng", "tím đầu chi", "đầu chi tím", "đầu ngón tím", "hoại tử", "vết loét",
          "sưng loét", "loét không lành", "đau kịch liệt", "tê chi", "tê chân", "tê tay")),
        (("tĩnh mạch viêm tắc", "viêm tắc tĩnh mạch", "tĩnh mạch viêm", "giãn tĩnh mạch"),
         ("chân", "bắp chân", "cẳng chân", "chi dưới", "bắp đùi", "gân xanh", "nổi gân",
          "giãn tĩnh mạch", "sợi mạch", "tĩnh mạch", "phù chân", "sưng chân", "đau chân")),
        (("long bế",),
         ("tiểu", "tiểu tiện", "đi tiểu", "bí tiểu", "tiểu bí", "bí đái", "đái", "nước tiểu",
          "tiểu không thông", "tiểu khó", "tiểu buốt", "tiểu rắt", "nhỏ giọt", "vô niệu", "niệu")),
        (("trưng hà", "trưng tích", "hà tích", "trưng khối"),
         ("khối u", "u cục", "khối", "cục", "hòn", "báng", "u xơ", "khối rắn", "sờ được khối",
          "hòn cục", "tích khối", "khối cứng", "nổi cục")),
        # Hoàng đản (vàng da) ĐỊNH NGHĨA bằng DA/CỦNG MẠC MẮT VÀNG — KHÔNG dùng 'nước tiểu vàng' làm
        # requires: nước tiểu vàng là dấu cực phổ biến, không đặc hiệu (cô đặc/mất nước), sẽ dán bệnh
        # vàng da cho ca hư hàn chỉ vì tiểu sẫm màu. Cả 5 thể hoàng đản trong CSV đều có da/mắt vàng
        # nên bỏ keyword này không mất self-recall.
        (("hoàng đản", "vàng da"),
         ("vàng da", "da vàng", "vàng mắt", "mắt vàng", "củng mạc vàng", "vàng củng mạc",
          "vàng bủng", "vàng tươi", "vàng đậm", "vàng xỉn", "vàng như", "sắc vàng", "hoàng đản")),
        (("tê bì", "ma mộc", "tê tay", "tê chân", "tê dại"),
         ("tê", "ma mộc", "châm chích", "kiến bò", "mất cảm giác")),
        (("thủy thũng", "phù thũng", "phù nề"),
         ("phù", "thũng", "sưng phù", "phù nề", "ấn lõm", "mọng nước", "mắt húp", "húp mặt")),
        (("ẩn chẩn", "mề đay", "mày đay", "phong chẩn"),
         ("mề đay", "mày đay", "mẩn", "ngứa", "sẩn", "ban đỏ", "nổi ban", "phát ban", "ẩn chẩn")),
        (("tử cung hạ sa", "sa tử cung", "âm đĩnh", "thoát giang", "sa trực tràng", "sa dạ dày",
          "vị hạ thùy", "vị hạ sa", "hạ sa"),
         ("sa tử cung", "tử cung sa", "khối sa", "sa xuống", "trằn nặng", "sa dạ con",
          "âm đĩnh", "lòi dom", "thoát giang", "sa trực tràng", "sa nội tạng", "sa dạ dày", "dạ dày sa")),
        (("tiêu khát", "đái tháo"),
         ("khát nước", "khát nhiều", "uống nhiều", "uống nước nhiều", "đa niệu", "tiểu nhiều",
          "đái nhiều", "tiểu tiện nhiều", "ăn nhiều", "mau đói", "chóng đói", "đói nhanh",
          "gầy sút", "sụt cân", "sút cân", "gầy nhiều", "tiêu khát", "đường huyết", "tiểu đường",
          "đái tháo")),
        (("mạch vành", "động mạch vành", "nhồi máu", "thiếu máu cơ tim", "xơ cứng động mạch",
          "xơ vữa", "tâm luật bất tề", "loạn nhịp", "rối loạn nhịp", "rung nhĩ", "tâm quý",
          "tim đập"),
         ("đau ngực", "đau thắt ngực", "tức ngực", "đau vùng tim", "đau trước tim", "vùng tim",
          "ngực", "vùng ngực", "nghẹt thở", "hồi hộp", "trống ngực", "đánh trống ngực", "tâm quý",
          "khó thở", "đau lan", "mạch vành", "nhồi máu", "loạn nhịp", "tim đập", "đập nhanh",
          "tim nhanh", "hụt hơi")),
        (("tỵ cứu", "tỵ uyên", "tỵ tắc", "viêm mũi", "viêm xoang", "viêm mũi xoang"),
         ("mũi", "hắt hơi", "sổ mũi", "chảy nước mũi", "chảy mũi", "ngạt mũi", "nghẹt mũi",
          "ngứa mũi", "tắc mũi", "nước mũi", "dịch mũi", "tỵ cứu", "tỵ uyên",
          "đau nhức vùng mặt", "nhức vùng mặt", "đau vùng má", "đau trán")),
        (("tỵ nục", "nục huyết", "chảy máu cam", "chảy máu mũi"),
         ("chảy máu cam", "chảy máu mũi", "máu cam", "máu mũi", "chảy máu", "xuất huyết",
          "nục huyết", "tỵ nục", "ra máu", "huyết ra", "ra đằng mũi", "đằng mũi", "sắc huyết")),
        (("bạo manh", "thanh manh", "dạ manh", "sắc manh"),
         ("mù", "lòa", "quáng gà", "mù màu", "mù lòa", "mất thị lực", "giảm thị lực", "thị lực giảm",
          "thị lực", "không nhìn thấy", "nhìn không rõ", "mờ mắt", "mắt mờ", "mắt kém", "màu sắc",
          "nhận biết màu", "sắc giác", "mắt có màng", "màng nổi", "đáy mắt", "mắt đau", "mắt tức",
          "bạo manh", "thanh manh", "dạ manh", "sắc manh")),
        (("cước căn thống", "cước căn"),
         ("gót chân", "đau gót", "gót", "cước căn", "đau bàn chân", "đau chân")),
        (("nuy chứng",),
         ("teo cơ", "teo", "liệt", "yếu chi", "chi yếu", "tay chân yếu", "chân tay yếu",
          "yếu hai chân", "yếu cơ", "nhược cơ", "bại liệt", "đi lại khó", "đi đứng khó",
          "mềm nhũn", "mềm yếu", "không cử động", "yếu liệt", "nuy chứng")),
        (("dương nuy",),
         ("liệt dương", "dương nuy", "rối loạn cương", "yếu sinh lý", "bất lực", "xuất tinh",
          "di tinh", "tinh trùng")),
        (("mất tiếng", "thất âm", "khản tiếng", "khàn tiếng"),
         ("mất tiếng", "khàn", "khản", "tắt tiếng", "nói không ra", "không nói được", "nói khó",
          "tiếng nói nặng", "tiếng nặng", "giọng", "thất âm", "mất giọng")),
        (("yếm thực", "biếng ăn", "chán ăn"),
         ("ăn kém", "biếng ăn", "chán ăn", "không muốn ăn", "ăn ít", "ăn uống kém", "yếm thực",
          "không thèm ăn", "nhìn thức ăn", "không thiết ăn", "bỏ ăn", "lười ăn")),
        (("tam thoa", "thần kinh tam thoa"),
         ("đau mặt", "mặt đau", "đau nửa mặt", "đau vùng mặt", "một bên mặt", "bên mặt",
          "đau nửa đầu", "một bên đầu", "đau như điện giật", "từng cơn", "co giật",
          "đau dây thần kinh", "tam thoa", "đau hàm", "đau má", "đau trán", "da mặt xám")),
        (("béo phì",),
         ("béo", "mập", "thừa cân", "tăng cân", "quá cân", "phát phì", "béo phì", "bụng to")),
        (("châm nhãn", "lẹo", "chắp", "kết mạc", "cam nhãn", "mạch nhãn", "cận thị"),
         ("mắt đỏ", "đỏ mắt", "mắt sưng", "sưng mắt", "đau mắt", "mắt đau", "nhức mắt", "cộm mắt",
          "ngứa mắt", "mắt ngứa", "chảy nước mắt", "mờ mắt", "mắt mờ", "khô mắt", "mỏi mắt",
          "mi mắt", "mí mắt", "bờ mi", "lẹo", "chắp", "kết mạc", "giác mạc", "nhặm", "ghèn",
          "nhìn mờ", "nhìn không rõ", "giảm thị lực", "con ngươi", "đồng tử", "tròng mắt")),
        (("canh niên", "mãn kinh", "tiền mãn kinh", "tuyệt kinh"),
         ("mãn kinh", "tắt kinh", "tuyệt kinh", "hết kinh", "sắp hết kinh", "kinh nguyệt",
          "rối loạn kinh", "kinh không đều", "hành kinh", "bốc hỏa", "cơn nóng bừng",
          "nóng bừng mặt", "trung niên", "đứng tuổi", "canh niên")),
        (("khẩu sang", "sang miệng"),
         ("loét miệng", "lở miệng", "nhiệt miệng", "miệng lở", "miệng lưỡi", "loét lưỡi",
          "vết loét", "nhiệt lưỡi", "khẩu sang", "sang miệng", "lở loét miệng",
          "tưa", "tưa lưỡi", "tưa miệng", "mảng trắng", "màng trắng", "mảng tưa", "đẹn")),
        (("nùng bào sang", "chốc lở"),
         ("tổn thương da", "mụn nước", "mụn mủ", "bọng mủ", "bọng nước", "nốt mủ", "chốc",
          "chốc lở", "lở loét", "da lở", "phỏng da", "mụn phỏng", "nùng bào", "ghẻ", "loét da")),
    ]
    for names, reqs in defining:
        gates.append({"names": list(names), "requires": list(reqs), "kw_wb": True})

    # ── Nhóm 0c: cổng ĐỊNH VỊ GIẢI PHẪU — substring/substring ──
    locus = {
        "âm hộ": ["âm hộ", "âm đạo", "vùng kín"],
        "âm đạo": ["âm đạo", "âm hộ", "vùng kín"],
        "tử cung": ["tử cung", "dạ con"],
        "buồng trứng": ["buồng trứng"],
        "tinh hoàn": ["tinh hoàn", "bìu"],
        "dương vật": ["dương vật"],
        "hậu môn": ["hậu môn", "lòi dom", "trĩ", "thoát giang"],
        "trực tràng": ["trực tràng", "hậu môn", "lòi dom"],
        "bàng quang": ["bàng quang", "tiểu buốt", "tiểu rắt", "tiểu khó", "bí tiểu", "tiểu ra máu", "tiểu nhiều lần"],
        "niệu đạo": ["niệu đạo", "tiểu buốt", "tiểu rắt"],
    }
    for loc, syns in locus.items():
        gates.append({"names": [loc], "requires": list(syns)})

    # ── Nhóm 1-24: cổng chuyên khoa ──
    gates += [
        {"names": ["trĩ"], "requires": ["trĩ", "hậu môn", "đại tiện ra máu", "tiêu ra máu",
                                        "đi ngoài ra máu", "ỉa ra máu", "sa búi", "búi trĩ"]},
        {"names": ["chấn thương sọ não"], "requires": ["chấn thương", "va đập", "tai nạn",
                                                       "ngã đầu", "đập đầu", "ngoại thương", "bị thương"]},
        {"names": ["phế quản", "viêm đường hô hấp"], "kw_wb": True,
         "requires": ["ho", "đờm", "đàm", "khạc", "họng", "cổ họng", "phế quản", "sổ mũi", "ngạt mũi",
                      "hắt hơi", "khó thở", "khò khè", "suyễn", "đoản khí", "hụt hơi", "tức ngực"]},
        {"names": ["viêm họng", "yết hầu", "hầu tý", "hầu phong", "nhũ nga", "khàn tiếng", "thất âm"], "kw_wb": True,
         "requires": ["họng", "hầu", "amidan", "khàn", "nuốt đau", "nuốt vướng", "rát cổ", "đau cổ",
                      "mất tiếng", "ho"]},
        {"names": ["xoang"], "requires": ["xoang", "mũi", "ngạt mũi", "chảy nước mũi", "sổ mũi", "tịt mũi"]},
        {"names": ["trúng phong", "tai biến", "đột quỵ"],
         "requires": ["liệt", "méo miệng", "bán thân bất toại", "tê bại", "khó nói", "mất ngôn ngữ", "trúng phong", "tai biến"]},
        {"names": ["addison", "alzheimer", "basedow", "parkinson", "eczema", "gout",
                   "tuyến thượng thận", "suy tim", "loãng xương",
                   "bệnh bạch huyết", "leukemia", "lymphoma", "lupus", "sclerosis",
                   "parathyroid", "cushing", "hashimoto", "hodgkin"],
         "requires": []},   # bệnh Tây y thuần -> luôn loại nếu khớp tên
        {"names": ["bạch huyết", "bạch cầu", "ung thư máu", "huyết hữu", "xuất huyết giảm tiểu cầu"],
         "requires": ["xuất huyết", "bầm tím tự phát", "chảy máu", "hạch", "gan lách to", "tiểu cầu"]},
        # 'nguyệt kinh'/'kinh nguyệt' là bigram AN TOÀN (không dính 'động kinh'/'thần kinh'/'kinh
        # phong') phủ họ rối loạn kinh nguyệt: Nguyệt kinh trì kỳ/tiên kỳ/quá đa, Kinh nguyệt rối
        # loạn — trước đây rơi ngoài cổng nên thể Hàn của chúng (sợ lạnh + tay chân lạnh + đau bụng)
        # bị gán cho ca tiêu hóa hư hàn không hề có triệu chứng kinh nguyệt.
        # 'khí hư' là BỆNH đới hạ/huyết trắng (完带汤 Hoàn đới thang) — ĐỒNG ÂM với hội chứng 'khí hư'
        # (khí hư nhược). Thể Tỳ hư của nó (mệt mỏi + tay chân lạnh + đại tiện lỏng) khớp bừa ca dương
        # hư nam, dán bệnh phụ khoa + bài trị đới hạ cho nam giới. Đòi hỏi bối cảnh phụ khoa (khí hư ra
        # nhiều/huyết trắng/đới hạ/âm đạo/kinh nguyệt...) mới cho qua. Cả 5 thể CSV đều mở đầu bằng
        # 'Khí hư ...' nên keyword 'khí hư' trong requires bảo toàn self-recall.
        {"names": ["lưu sản", "sảy thai", "băng lậu", "đới hạ", "vô sinh", "bế kinh", "thống kinh",
                   "sản hậu", "nhau thai", "thai chết", "động thai", "an thai", "thai lậu", "hoạt thai",
                   "nguyệt kinh", "kinh nguyệt", "khí hư"],
         "requires": ["kinh nguyệt", "kinh", "thai", "sản", "âm đạo", "huyết trắng", "đới hạ",
                      "băng", "lậu", "tử cung", "phụ nữ", "mang thai", "có thai", "khí hư"]},
        {"names": ["nhĩ minh", "nhĩ lung", "điếc", "ù tai", "viêm tai"],
         "requires": ["ù tai", "tai", "điếc", "nghe kém", "nhĩ minh", "nhĩ lung"]},
        {"names": ["thanh manh", "nhược thị", "mắt mờ", "quáng gà", "đục thủy tinh", "nội chướng", "ngoại chướng"],
         "requires": ["mắt mờ", "nhìn mờ", "thị lực", "quáng gà", "mắt đau", "mắt sưng"]},
        {"names": ["mề đay", "chàm", "ghẻ", "hắc lào", "vẩy nến", "mụn nhọt",
                   "ung nhọt", "ngân tiết", "bạch bì", "tùng bì tiễn", "vảy nến", "bạch tiển"],
         "requires": ["ngứa", "mẩn", "ban", "mụn", "ghẻ", "da", "phát ban", "mề đay", "nổi mề",
                      "vảy", "bong vảy", "tổn thương da", "mảng", "sẩn", "lở", "loét da"]},
        {"names": ["đái dầm", "di niệu", "sỏi thận", "sỏi tiết niệu", "viêm bàng quang"],
         "requires": ["tiểu", "đái", "niệu", "bàng quang", "sỏi", "tiểu đêm", "tiểu gắt"]},
        {"names": ["viêm cầu thận", "suy thận", "thận hư hội chứng", "viêm thận", "hội chứng thận hư"],
         "requires": ["phù", "tiểu ít", "tiểu đục", "tiểu ra máu", "phù mặt", "phù chân",
                      "albumin", "protein niệu", "thận"]},
        {"names": ["ung thư", "u ác", "khối u", "nhục lựu"], "names_regex": ["ái"],
         "requires": ["khối u", "sụt cân", "nuốt nghẹn", "ho ra máu", "u bướu", "sưng hạch", "di căn", "ung thư"]},
        {"names": ["thực quản", "vị quản", "loét dạ dày", "viêm dạ dày"],
         "requires": ["đau bụng", "đau dạ dày", "ợ chua", "ợ hơi", "buồn nôn", "nôn", "nuốt nghẹn", "trào ngược", "thượng vị"]},
        {"names": ["dương nuy", "liệt dương", "di tinh", "tảo tiết", "hoạt tinh", "dương sự",
                   "mộng tinh", "âm hành", "cao hoàn", "âm nang"],
         "requires": ["liệt dương", "dương nuy", "di tinh", "mộng tinh", "tảo tiết", "hoạt tinh",
                      "xuất tinh", "dương vật", "rối loạn cương", "cương dương", "yếu sinh lý",
                      "sinh lý", "tình dục", "âm hành", "cao hoàn", "âm nang", "bìu"]},
        {"names": ["hung tý", "hung tí", "chân tâm thống", "tâm thống", "tâm giảo thống", "nhồi máu", "tâm nhồi máu"],
         "requires": ["đau ngực", "tức ngực", "ngực đau", "đau vùng ngực", "đau thắt ngực",
                      "ngực đầy", "đầy tức ngực", "đau trước tim", "đau tim", "đau thắt tim", "hung tý",
                      "hồi hộp", "trống ngực", "tim đập"]},
        {"names": ["quỷ thai", "chửa trứng", "thai trứng", "chửa trâu"],
         "requires": ["có thai", "mang thai", "thai nghén", "thai động", "que thử thai", "chửa", "ốm nghén", "thai lưu"]},
        {"names": ["thiên đầu thống", "đầu thống", "đau nửa đầu"],
         "requires": ["đau đầu", "nhức đầu", "đau nửa đầu", "nặng đầu", "đầu đau", "váng đầu", "đau vùng đầu", "đau nhức đầu"]},
        {"names": ["tâm quý", "chinh xung", "kinh quý", "tâm quí", "đánh trống ngực"],
         "requires": ["hồi hộp", "trống ngực", "tim đập", "đánh trống ngực", "tim hồi hộp", "loạn nhịp",
                      "tim đập nhanh", "kinh sợ", "hoảng sợ", "dễ giật mình", "tâm quý"]},
        {"names": ["ma chẩn", "phong chẩn", "thủy đậu", "thuỷ đậu", "ban chẩn", "đơn độc", "phong ngứa"],
         "requires": ["ban", "phát ban", "nổi ban", "mọc ban", "mụn nước", "nốt", "hồng ban", "sởi", "phỏng", "ngứa", "mẩn"]},
        {"names": ["nhũ ung", "nhũ nham", "nhũ phích", "nhũ lạc", "viêm tuyến vú", "nhũ tuyến", "nhũ"],
         "requires": ["vú", "tuyến vú", "đau vú", "sưng vú", "cục ở vú", "núm vú", "tắc sữa", "áp xe vú"]},
        {"names": ["áp xe phế", "phế ung", "áp xe phổi"],
         "requires": ["ho ra mủ", "khạc mủ", "ho ra máu", "đờm mủ", "đờm tanh", "đau ngực", "mủ tanh", "sốt cao rét run", "khạc ra máu"]},
        {"names": ["phát nhiệt", "phát sốt"],
         "requires": ["sốt", "phát nhiệt", "phát sốt", "triều nhiệt", "cốt chưng", "ngũ tâm phiền nhiệt",
                      "nóng trong", "sốt về chiều", "hâm hấp", "nóng về chiều", "sốt nhẹ", "hầm hập", "nhiệt độ"]},

        # ── Nhóm 25: cổng bệnh CHUYÊN KHOA còn thiếu (rà bằng scripts + quét leak trên profile
        # triệu chứng CHUNG). Trước đây các bệnh định vị tạng/hệ này KHÔNG có cổng nên lọt vào ca
        # thể trạng chung chỉ nhờ dấu phụ (vd COPD dán ca MẤT NGỦ vì trùng 'sợ gió/tự hãn' của thể
        # Phế khí hư). requires phủ đủ triệu chứng ĐỊNH NGHĨA của mọi thể CSV -> bảo toàn self-recall.
        # (Bỏ qua Nhiễm mỡ xơ mạch — triệu chứng CSV toàn chóng mặt/váng đầu chung, không có từ khóa
        # mỡ/xơ vữa nên không gate được mà không phá self-recall; Nhi bại liệt — giai đoạn đầu giống
        # sốt virus; Suy nhược thần kinh — mệt/mất ngủ vốn LÀ bệnh cảnh đúng của nó.)
        {"names": ["tắc nghẽn phế", "phổi tắc nghẽn", "mạn tính tắc nghẽn"], "kw_wb": True,
         "requires": ["ho", "đờm", "đàm", "khạc", "khó thở", "khò khè", "suyễn", "đoản khí", "hụt hơi",
                      "tức ngực", "thở", "phổi", "phế", "hô hấp"]},
        {"names": ["hung thống"],
         "requires": ["đau ngực", "tức ngực", "ngực đau", "đau vùng ngực", "xương ức", "sau xương ức",
                      "đau thắt ngực", "ngực đầy", "đau lan", "hồi hộp", "trống ngực", "vai"]},
        {"names": ["hầu ung", "hạnh đào"], "kw_wb": True,
         "requires": ["họng", "hầu", "amidan", "hạnh đào", "nuốt", "rát cổ", "đau cổ", "sưng họng",
                      "họng đau", "họng sưng", "ho"]},
        {"names": ["đại tràng", "trường ung", "ruột thừa"],
         "requires": ["đau bụng", "bụng đau", "bụng dưới", "quanh rốn", "tiêu chảy", "đại tiện", "phân",
                      "đi ngoài", "đi lỏng", "ỉa", "táo bón", "sôi bụng", "đầy bụng", "ruột", "cự án"]},
        {"names": ["áp xe gan"],
         "requires": ["hạ sườn", "sườn phải", "hông phải", "hông sườn", "vùng gan", "gan", "sốt",
                      "nóng lạnh", "vàng da", "áp xe", "mủ", "đau sườn"]},
        {"names": ["thận vu viêm", "bể thận", "thận quặn"],
         "requires": ["thận", "tiểu", "đái", "niệu", "đau lưng", "thắt lưng", "đau quặn", "sỏi", "hông",
                      "phù", "sốt", "vùng thận"]},
        {"names": ["khớp dạng thấp", "viêm khớp", "cốt lao", "lao xương"],
         "requires": ["khớp", "đau khớp", "sưng khớp", "cứng khớp", "viêm khớp", "xương khớp", "đau xương",
                      "nhức xương", "khớp xương", "teo cơ", "biến dạng", "đau nhức", "sưng đau"]},
        {"names": ["bắp chân xung đau"],
         "requires": ["bắp chân", "chân", "chi dưới", "cẳng chân", "sưng chân", "đau chân", "tĩnh mạch",
                      "sợi mạch", "gân"]},
        {"names": ["bạch biến", "ngoan tiển"],
         "requires": ["da", "vết trắng", "đốm trắng", "mất sắc tố", "bạch biến", "ngứa", "tổn thương da",
                      "sẩn", "da dày", "mảng da", "vảy"]},
    ]

    payload = {
        "version": 1,
        "description": "Cổng an toàn bệnh danh (data-hóa từ _validate_disease_safety). names: khớp tên "
                       "bệnh (substring); names_regex: khớp tên theo ranh giới từ; requires: lời khai "
                       "phải có >=1 từ (rỗng = luôn loại — bệnh Tây y thuần); kw_wb: khớp requires theo "
                       "ranh giới từ. Sinh bởi scripts/build_disease_gates.py.",
        "gates": gates,
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Đã ghi {len(gates)} cổng -> {OUT}")


if __name__ == "__main__":
    build()
