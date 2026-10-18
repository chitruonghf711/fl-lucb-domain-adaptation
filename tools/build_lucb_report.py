"""Build the Vietnamese technical LUCB change report as a DOCX artifact."""

from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.table import WD_ALIGN_VERTICAL
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Mm, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "Bao_cao_thay_doi_ky_thuat_LUCB.docx"

HEADER_FILL = "E7E6E6"
ALT_FILL = "F7F7F7"
BORDER = "808080"
BLACK = RGBColor(0, 0, 0)


def set_cell_shading(cell, fill: str) -> None:
    properties = cell._tc.get_or_add_tcPr()
    shading = properties.find(qn("w:shd"))
    if shading is None:
        shading = OxmlElement("w:shd")
        properties.append(shading)
    shading.set(qn("w:fill"), fill)
    shading.set(qn("w:val"), "clear")


def set_cell_border(cell, color: str = BORDER) -> None:
    properties = cell._tc.get_or_add_tcPr()
    borders = properties.first_child_found_in("w:tcBorders")
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        properties.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        tag = qn(f"w:{edge}")
        element = borders.find(tag)
        if element is None:
            element = OxmlElement(f"w:{edge}")
            borders.append(element)
        element.set(qn("w:val"), "single")
        element.set(qn("w:sz"), "6")
        element.set(qn("w:space"), "0")
        element.set(qn("w:color"), color)


def clear_cell_borders(cell) -> None:
    properties = cell._tc.get_or_add_tcPr()
    borders = properties.first_child_found_in("w:tcBorders")
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        properties.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        tag = qn(f"w:{edge}")
        element = borders.find(tag)
        if element is None:
            element = OxmlElement(f"w:{edge}")
            borders.append(element)
        element.set(qn("w:val"), "nil")


def clear_table_borders(table) -> None:
    for row in table.rows:
        for cell in row.cells:
            clear_cell_borders(cell)


def set_paragraph_bottom_border(paragraph, color: str = "000000", size: str = "8") -> None:
    properties = paragraph._p.get_or_add_pPr()
    borders = properties.find(qn("w:pBdr"))
    if borders is None:
        borders = OxmlElement("w:pBdr")
        properties.append(borders)
    bottom = borders.find(qn("w:bottom"))
    if bottom is None:
        bottom = OxmlElement("w:bottom")
        borders.append(bottom)
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), size)
    bottom.set(qn("w:space"), "1")
    bottom.set(qn("w:color"), color)


def set_cell_margins(cell, top: int = 50, start: int = 100, bottom: int = 50, end: int = 100) -> None:
    properties = cell._tc.get_or_add_tcPr()
    margins = properties.first_child_found_in("w:tcMar")
    if margins is None:
        margins = OxmlElement("w:tcMar")
        properties.append(margins)
    for side, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = margins.find(qn(f"w:{side}"))
        if node is None:
            node = OxmlElement(f"w:{side}")
            margins.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def set_repeat_table_header(row) -> None:
    properties = row._tr.get_or_add_trPr()
    marker = OxmlElement("w:tblHeader")
    marker.set(qn("w:val"), "true")
    properties.append(marker)


def prevent_row_split(row) -> None:
    properties = row._tr.get_or_add_trPr()
    marker = OxmlElement("w:cantSplit")
    properties.append(marker)


def set_column_widths(table, widths: list[float]) -> None:
    for row in table.rows:
        for index, width in enumerate(widths):
            row.cells[index].width = Inches(width)


def format_table(table, widths: list[float] | None = None) -> None:
    table.style = "Table Grid"
    table.autofit = False
    if widths is not None:
        set_column_widths(table, widths)
    for row_index, row in enumerate(table.rows):
        prevent_row_split(row)
        if row_index == 0:
            set_repeat_table_header(row)
        for cell in row.cells:
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            set_cell_border(cell)
            set_cell_margins(cell)
            set_cell_shading(cell, HEADER_FILL if row_index == 0 else (ALT_FILL if row_index % 2 == 0 else "FFFFFF"))
            for paragraph in cell.paragraphs:
                paragraph.paragraph_format.space_after = Pt(0)
                paragraph.paragraph_format.space_before = Pt(0)
                paragraph.paragraph_format.line_spacing = 1.08
                for run in paragraph.runs:
                    run.font.name = "Times New Roman"
                    run._element.rPr.rFonts.set(qn("w:ascii"), "Times New Roman")
                    run._element.rPr.rFonts.set(qn("w:hAnsi"), "Times New Roman")
                    run._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
                    run.font.size = Pt(10.5)
                    run.font.color.rgb = BLACK
                    run.bold = row_index == 0


def add_table(document: Document, headers: list[str], rows: list[list[str]], widths: list[float]) -> None:
    table = document.add_table(rows=1, cols=len(headers))
    for index, value in enumerate(headers):
        table.cell(0, index).text = value
    for values in rows:
        cells = table.add_row().cells
        for index, value in enumerate(values):
            cells[index].text = value
    format_table(table, widths)
    document.add_paragraph().paragraph_format.space_after = Pt(1)


def set_run_font(run, size: float = 10.2, bold: bool = False, color: RGBColor = BLACK) -> None:
    run.font.name = "Times New Roman"
    run._element.rPr.rFonts.set(qn("w:ascii"), "Times New Roman")
    run._element.rPr.rFonts.set(qn("w:hAnsi"), "Times New Roman")
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = False
    run.font.color.rgb = color


def add_paragraph(
    document: Document,
    text: str = "",
    *,
    bold_lead: str | None = None,
    justify: bool = True,
) -> None:
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY if justify else WD_ALIGN_PARAGRAPH.LEFT
    paragraph.paragraph_format.first_line_indent = Mm(10)
    paragraph.paragraph_format.space_after = Pt(6)
    paragraph.paragraph_format.line_spacing = 1.15
    if bold_lead and text.startswith(bold_lead):
        lead = paragraph.add_run(bold_lead)
        set_run_font(lead, size=13, bold=True)
        rest = paragraph.add_run(text[len(bold_lead) :])
        set_run_font(rest, size=13)
    else:
        run = paragraph.add_run(text)
        set_run_font(run, size=13)


def add_bullet(document: Document, text: str, level: int = 0) -> None:
    style = "List Bullet" if level == 0 else "List Bullet 2"
    paragraph = document.add_paragraph(style=style)
    paragraph.paragraph_format.space_after = Pt(3)
    paragraph.paragraph_format.line_spacing = 1.15
    run = paragraph.add_run(text)
    set_run_font(run, size=13)


def add_heading(document: Document, text: str, level: int = 1) -> None:
    paragraph = document.add_heading(text, level=level)
    paragraph.paragraph_format.space_before = Pt(12 if level == 1 else 8)
    paragraph.paragraph_format.space_after = Pt(6)
    paragraph.paragraph_format.keep_with_next = True
    for run in paragraph.runs:
        set_run_font(run, size=14 if level == 1 else 13, bold=True)


def set_default_styles(document: Document) -> None:
    normal = document.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal._element.rPr.rFonts.set(qn("w:ascii"), "Times New Roman")
    normal._element.rPr.rFonts.set(qn("w:hAnsi"), "Times New Roman")
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    normal.font.size = Pt(13)
    normal.font.color.rgb = BLACK
    for style_name in ("Title", "Subtitle", "Heading 1", "Heading 2"):
        style = document.styles[style_name]
        style.font.name = "Times New Roman"
        style._element.rPr.rFonts.set(qn("w:ascii"), "Times New Roman")
        style._element.rPr.rFonts.set(qn("w:hAnsi"), "Times New Roman")
        style._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
        style.font.color.rgb = BLACK
    title_properties = document.styles["Title"]._element.get_or_add_pPr()
    title_border = title_properties.find(qn("w:pBdr"))
    if title_border is not None:
        title_properties.remove(title_border)


def add_page_number(section) -> None:
    paragraph = section.header.paragraphs[0]
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.space_after = Pt(0)
    run = paragraph.add_run()
    set_run_font(run, size=13)
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instruction = OxmlElement("w:instrText")
    instruction.set(qn("xml:space"), "preserve")
    instruction.text = " PAGE "
    separate = OxmlElement("w:fldChar")
    separate.set(qn("w:fldCharType"), "separate")
    text = OxmlElement("w:t")
    text.text = "1"
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    run._r.extend((begin, instruction, separate, text, end))


def configure_document(document: Document) -> None:
    properties = document.core_properties
    properties.title = "Báo cáo kỹ thuật LUCB minibatch"
    properties.subject = "Phát triển LUCB minibatch cho Federated Learning đa miền"
    properties.author = ""
    properties.last_modified_by = ""
    properties.comments = ""
    properties.keywords = "LUCB, Federated Learning, technical report"
    properties.category = "Báo cáo kỹ thuật"
    section = document.sections[0]
    section.page_width = Mm(210)
    section.page_height = Mm(297)
    section.top_margin = Mm(25)
    section.bottom_margin = Mm(20)
    section.left_margin = Mm(30)
    section.right_margin = Mm(20)
    section.header_distance = Mm(10)
    section.footer_distance = Mm(10)
    section.different_first_page_header_footer = True
    add_page_number(section)
    set_default_styles(document)


def add_administrative_heading(document: Document) -> None:
    table = document.add_table(rows=1, cols=2)
    table.autofit = False
    set_column_widths(table, [3.05, 3.05])
    clear_table_borders(table)
    left, right = table.rows[0].cells

    left_paragraph = left.paragraphs[0]
    left_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    left_paragraph.paragraph_format.space_after = Pt(1)
    left_run = left_paragraph.add_run("[CƠ QUAN CHỦ QUẢN]\n[ĐƠN VỊ CHỦ TRÌ]")
    set_run_font(left_run, size=12, bold=True)
    set_paragraph_bottom_border(left_paragraph)
    number_paragraph = left.add_paragraph()
    number_paragraph.paragraph_format.space_after = Pt(0)
    number_run = number_paragraph.add_run("Số: [......]/BC-[......]")
    set_run_font(number_run, size=12)

    right_paragraph = right.paragraphs[0]
    right_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    right_paragraph.paragraph_format.space_after = Pt(1)
    right_run = right_paragraph.add_run("CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM\nĐộc lập - Tự do - Hạnh phúc")
    set_run_font(right_run, size=13, bold=True)
    set_paragraph_bottom_border(right_paragraph)

    date_paragraph = document.add_paragraph()
    date_paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    date_paragraph.paragraph_format.space_before = Pt(5)
    date_paragraph.paragraph_format.space_after = Pt(12)
    date_run = date_paragraph.add_run("[Địa danh], ngày 18 tháng 10 năm 2026")
    set_run_font(date_run, size=13)
    date_run.italic = True

    title = document.add_paragraph(style="Title")
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.space_after = Pt(4)
    title_run = title.add_run("BÁO CÁO KỸ THUẬT")
    set_run_font(title_run, size=14, bold=True)

    summary = document.add_paragraph(style="Subtitle")
    summary.alignment = WD_ALIGN_PARAGRAPH.CENTER
    summary.paragraph_format.space_after = Pt(16)
    summary_run = summary.add_run("Về việc phát triển LUCB minibatch cho Federated Learning đa miền")
    set_run_font(summary_run, size=14, bold=True)


def add_signature_block(document: Document) -> None:
    document.add_paragraph().paragraph_format.space_after = Pt(4)
    table = document.add_table(rows=1, cols=2)
    table.autofit = False
    set_column_widths(table, [3.1, 3.0])
    clear_table_borders(table)
    recipients, signer = table.rows[0].cells

    recipient_paragraph = recipients.paragraphs[0]
    recipient_paragraph.paragraph_format.space_after = Pt(2)
    recipient_label = recipient_paragraph.add_run("Nơi nhận:")
    set_run_font(recipient_label, size=12, bold=True)
    for line in ("- [Đơn vị nhận báo cáo];", "- Lưu: Hồ sơ dự án."):
        paragraph = recipients.add_paragraph()
        paragraph.paragraph_format.space_after = Pt(1)
        run = paragraph.add_run(line)
        set_run_font(run, size=12)

    signer_paragraph = signer.paragraphs[0]
    signer_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    signer_paragraph.paragraph_format.space_after = Pt(4)
    signer_run = signer_paragraph.add_run("NGƯỜI LẬP BÁO CÁO")
    set_run_font(signer_run, size=13, bold=True)
    instruction_paragraph = signer.add_paragraph()
    instruction_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    instruction_paragraph.paragraph_format.space_after = Pt(35)
    instruction_run = instruction_paragraph.add_run("(Ký, ghi rõ họ tên)")
    set_run_font(instruction_run, size=12)
    instruction_run.italic = True
    name_paragraph = signer.add_paragraph()
    name_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    name_run = name_paragraph.add_run("[Họ và tên]")
    set_run_font(name_run, size=13, bold=True)


def add_report_body(document: Document) -> None:
    add_heading(document, "1 Kết luận và phạm vi")
    add_paragraph(
        document,
        "Triển khai mới thay luồng đánh giá toàn bộ model client bằng bộ chọn LUCB dùng "
        "minibatch có ngân sách rõ ràng. Cơ chế dùng error rate để xếp client yếu, dùng "
        "Wilson interval để quyết định lớp yếu, và chỉ tạo feedback khi kết quả LUCB đã hoàn tất "
        "và có lớp yếu được kết luận. Mặc định là abstract feedback; legacy raw feedback phải bật "
        "rõ ràng bằng allow_raw_feedback."
    )
    add_paragraph(
        document,
        "Baseline để đối chiếu là commit ee7ec48df9bd84cfb4b9b29f7fadc3174fcf233c. Các thư mục "
        "result_tabular và result_text lịch sử không bị sửa. Các artefact minibatch cũ trong "
        "result_tabular không được dùng làm bằng chứng tái lập vì không bảo đảm khớp hoàn toàn với "
        "source baseline hiện tại."
    )
    add_bullet(document, "Phạm vi chạy end-to-end trong đợt này: tabular Flower 5 client, 2 rounds, abstract feedback.")
    add_bullet(document, "Phạm vi xác nhận text/image: static compile, schema, CLI/startup validation và unit/stub tests.")
    add_bullet(document, "Không tuyên bố text/image đã chạy end-to-end vì thiếu checkpoint, GPU, token hoặc dataset runtime phù hợp.")

    add_heading(document, "2 Kiến trúc dùng chung")
    add_paragraph(
        document,
        "Package fl_lucb không phụ thuộc Torch hoặc Flower. Nó tách lập kế hoạch minibatch, "
        "tích lũy evidence, LUCB selection, Wilson intervals, schema feedback và audit sanitization "
        "khỏi từng domain. Điều này giữ logic ngân sách và privacy nhất quán giữa tabular, text và image."
    )
    add_table(
        document,
        ["Khối", "Vai trò", "Bảo đảm chính"],
        [
            ["fl_lucb/minibatch.py", "Lập schedule theo client", "Best-effort stratified, seed cố định, không lặp sample trong plan"],
            ["fl_lucb/wilson.py", "Khoảng tin cậy Wilson", "Đánh giá error và accuracy lớp bằng evidence đã quan sát"],
            ["fl_lucb/selector.py", "LUCB và legacy selector", "M đếm model-sample evaluations, không vượt budget"],
            ["fl_lucb/protocols.py", "Schema feedback", "Abstract loại bỏ raw fields; legacy cần opt-in"],
            ["fl_lucb/audit.py", "Audit metadata-only", "Loại raw row, token, pixel, tensor, reference trước khi ghi JSON"],
        ],
        [1.58, 1.72, 3.15],
    )
    add_paragraph(
        document,
        "Luồng selection: lập plan không lặp mẫu -> warm-up một batch cho mỗi client -> chọn cặp "
        "boundary arm -> chấm batch kế tiếp trong ngân sách -> dừng khi confidence separated, epsilon reached, "
        "all batches exhausted, hoặc budget exhausted. Client có error cao hơn được xem là yếu hơn.",
        justify=False,
    )

    add_heading(document, "3 Thống kê và chính sách dừng")
    add_paragraph(
        document,
        "Một lớp có accuracy Wilson upper bound nhỏ hơn tau được gắn weak; lower bound lớn hơn tau là "
        "adequate; còn lại là uncertain. Phiên bản này phân bổ lucb_delta theo finite-horizon Bonferroni "
        "trên toàn bộ error looks và class looks đã lên kế hoạch. Cách này thận trọng hơn việc dùng cùng một "
        "delta cho từng lần chấm adaptive.",
        justify=False,
    )
    add_table(
        document,
        ["Điều kiện", "Hành vi"],
        [
            ["Weights/fingerprint client thay đổi", "strict reset xóa evidence Wilson; không dùng evidence cũ cho model mới"],
            ["decay_priority", "Chỉ giữ priority lịch chọn; evidence Wilson vẫn reset khi fingerprint thay đổi"],
            ["Budget không đủ warm-up", "Trả trạng thái insufficient, không gọi scorer"],
            ["Budget exhausted hoặc partial", "Không gửi feedback; audit ghi lý do không delivery"],
            ["Không có weak class kết luận", "Không ép tạo feedback, audit ghi no_confident_weak_class"],
        ],
        [2.05, 4.4],
    )

    add_heading(document, "4 Mapping mã gốc sang mã mới")
    add_table(
        document,
        ["Điểm baseline", "Mã mới", "Thay đổi có thể đối chiếu"],
        [
            ["Không có selector dùng chung", "fl_lucb/", "Tách LUCB, Wilson, plan, protocol và audit thành package thuần Python"],
            ["tabular/server_tabular.py full scan", "tabular/server_tabular.py", "Callback LUCB chấm minibatch; SHAP chỉ trên lỗi đã quan sát của client/lớp chọn"],
            ["tabular/client_tabular.py", "tabular/client_tabular.py", "Payload abstract điều khiển TVAE cục bộ bằng label yếu và rank feature tổng hợp"],
            ["text/server_textdata.py", "text/server_textdata.py", "Path CLI ổn định, LUCB, Grad-CAM trên observed errors, audit và routing feedback"],
            ["text/client_textdata.py", "text/client_textdata.py", "Stable client ID, label-map fingerprint handshake, abstract prompt theo label/position"],
            ["image/client_1..9.py", "image/client_1.py + wrappers", "Một runtime chung; client_2..9 chỉ giữ entrypoint tương thích"],
            ["image/server.py full accuracy selection", "image/server.py", "LUCB, Grad-CAM chỉ trên observed errors, audit legacy/abstract và validation mode"],
            ["Không có test LUCB độc lập", "tests/test_*.py", "Kiểm thử selector, privacy, track integration, image stub và dependency pins"],
        ],
        [1.9, 1.8, 2.75],
    )

    add_heading(document, "5 Cấu hình thống nhất")
    add_table(
        document,
        ["Khóa", "Pilot default", "Ý nghĩa"],
        [
            ["selection_mode", "lucb", "LUCB bounded hoặc legacy_full_scan cho baseline"],
            ["lucb_budget_fraction", "0.25", "Tỷ lệ chi phí full client scan dành cho M"],
            ["lucb_batch_size", "64", "Kích thước minibatch trong plan"],
            ["lucb_top_k", "auto, tối đa 3", "Số client yếu được giữ lại"],
            ["lucb_epsilon", "0.02", "Biên dừng LUCB"],
            ["lucb_delta", "0.05", "Family-wise Wilson error budget trên planned looks"],
            ["weak_class_accuracy", "0.85", "Ngưỡng tau để kết luận lớp weak"],
            ["seed", "42", "Plan/tie-breaking deterministic"],
            ["lucb_prioritization", "strict", "strict reset hoặc experimental decay_priority"],
            ["feedback_protocol", "abstract", "Schema privacy-safe mặc định"],
        ],
        [1.62, 1.36, 3.47],
    )
    add_paragraph(
        document,
        "M chỉ gồm các model-sample evaluations của selection client. Global-model monitoring được ghi "
        "riêng trong global_monitoring.jsonl và có trường included_in_lucb_budget=false; nó không được nhập vào M.",
        justify=False,
    )

    add_heading(document, "6 Feedback và quyền riêng tư")
    add_table(
        document,
        ["Domain", "Abstract mặc định", "Legacy có điều kiện"],
        [
            ["Tabular", "weak_labels + aggregate SHAP feature_ranking", "Raw/reference chỉ khi feedback_protocol=legacy và allow_raw_feedback"],
            ["Text", "weak_labels + phân bố vị trí beginning/middle/end", "Token feedback chỉ khi opt-in; audit vẫn loại token"],
            ["Image", "weak_labels + grid/patch coordinates, mode refine", "Raw patch values chỉ khi opt-in; abstract reject mode khác refine"],
        ],
        [0.88, 2.75, 2.82],
    )
    add_paragraph(
        document,
        "Audit JSON cho mọi feedback event ghi configuration hiệu lực, model fingerprint, M, budget, confidence bounds, "
        "coverage, selected clients/classes, boundary gap, stop reason và explanation_cost. Audit sanitizer luôn loại dữ liệu "
        "thô, kể cả runtime legacy được bật có chủ đích.",
        justify=False,
    )
    add_bullet(document, "Tabular: server không gửi target row/reference/feature value ở abstract mode; TVAE dùng thống kê local để ưu tiên vùng thưa.")
    add_bullet(document, "Text: prompt chỉ chứa label và vị trí salient; client kiểm tra fingerprint/num_labels trước khi train.")
    add_bullet(document, "Image: Grad-CAM chạy trên observed errors của chính selected client; abstract dùng local same-label canvas hoặc class-conditional fallback.")

    add_heading(document, "7 Tương thích legacy và tính lặp lại")
    add_paragraph(
        document,
        "selection_mode=legacy_full_scan giữ đường quét toàn bộ cho baseline thí nghiệm. Legacy không phải default và "
        "không được xem là privacy-preserving. Khi feedback_protocol=legacy, cả server/client yêu cầu allow_raw_feedback và "
        "phát warning; audit vẫn sanitized. Image legacy ghi audit chuẩn hóa với full-scan cost và explanation cost.",
        justify=False,
    )
    add_table(
        document,
        ["Mục tiêu", "Lệnh hoặc thao tác"],
        [
            ["So sánh thay đổi tổng thể", "git diff --stat ee7ec48"],
            ["So sánh selector chung", "git diff ee7ec48 -- fl_lucb tests"],
            ["So sánh tabular", "git diff ee7ec48 -- tabular/server_tabular.py tabular/client_tabular.py"],
            ["So sánh text", "git diff ee7ec48 -- text/server_textdata.py text/client_textdata.py"],
            ["So sánh image", "git diff ee7ec48 -- image/server.py image/client_1.py image/client_2.py"],
            ["Chạy baseline legacy", "python server_tabular.py --selection-mode legacy_full_scan --feedback-protocol legacy --allow-raw-feedback"],
        ],
        [2.0, 4.45],
    )

    add_heading(document, "8 Kiểm thử và kết quả thực thi")
    add_table(
        document,
        ["Hạng mục", "Kết quả", "Ghi chú"],
        [
            ["Unit tests", "33 passed", "Wilson/LUCB, budget, determinism, rare class, strict reset, audit/schema, domain integration"],
            ["Static compile", "Passed", "compileall cho fl_lucb, tabular, text, image và tests"],
            ["Tabular Flower smoke", "Passed", "5 clients, 2 rounds, abstract feedback, output run riêng"],
            ["Smoke budget", "M=320 <= 3750", "round 1 status=ok, stop_reason=epsilon_reached"],
            ["Smoke privacy", "Passed", "3 abstract feedback payloads; script từ chối raw/reference/value-style fields"],
            ["Text startup", "Passed", "server/client CLI help và dependency-light entrypoint tests"],
            ["Image startup/schema", "Passed", "CLI help + validate-config trong isolated environment; stub/dependency tests"],
        ],
        [1.48, 1.4, 3.55],
    )
    add_paragraph(
        document,
        "Smoke run dùng ngưỡng test-only lucb_epsilon=1.0 và weak_class_accuracy=1.0 để chủ động kích hoạt feedback; "
        "pilot defaults trong mã vẫn là 0.02 và 0.85. Artefact smoke thành công nằm tại "
        "tabular/logs/tabular_lucb_smoke_20261018_161917 và không đụng tới result_*.",
        justify=False,
    )

    add_heading(document, "9 Lệnh tái lập")
    add_table(
        document,
        ["Mục đích", "Lệnh"],
        [
            ["Toàn bộ test", "python -B -m unittest discover -s tests -p test_*.py -v"],
            ["Static compile", "python -m compileall -q fl_lucb tabular text image tests"],
            ["Tabular selector smoke", "python tabular/smoke_lucb.py"],
            ["Tabular Flower smoke", "powershell -ExecutionPolicy Bypass -File tabular/run_smoke_lucb.ps1"],
            ["Text LUCB server", "python text/server_textdata.py --selection-mode lucb --feedback-protocol abstract"],
            ["Image config validation", "python image/server.py --target <target-subset> --validate-config --mode refine"],
        ],
        [1.6, 4.83],
    )
    add_paragraph(
        document,
        "Image requirements pin Flower 1.8.0, protobuf 4.25.3, NumPy 1.26.4, Torch 2.2.2 và TorchVision 0.17.2. "
        "Cần tạo environment riêng theo pins này; không trộn với host TensorFlow/Flower đã nâng NumPy hoặc protobuf.",
        justify=False,
    )

    add_heading(document, "10 Giới hạn đã ghi nhận")
    add_bullet(document, "Bonferroni finite-horizon bảo thủ; một feedback event hợp lệ có thể hết budget hoặc chưa kết luận được lớp yếu. Đây là trạng thái có chủ đích, không phải lỗi fallback.")
    add_bullet(document, "Text full end-to-end cần DistilBERT assets, năm client, và token/quyền truy cập nếu dùng Gemma generation.")
    add_bullet(document, "Image vẫn experimental và cần CIFAR subset folders, cifar_diffusion.pth, cùng GPU/phù hợp binary stack. Không có claim image end-to-end trong báo cáo này.")
    add_bullet(document, "Trong abstract image mode chỉ hỗ trợ refine; cấu hình repaint/paper bị từ chối rõ ràng.")


def build_report() -> Path:
    document = Document()
    configure_document(document)
    if document.paragraphs:
        initial_paragraph = document.paragraphs[0]
        initial_paragraph._element.getparent().remove(initial_paragraph._element)
    add_administrative_heading(document)
    add_report_body(document)
    add_signature_block(document)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    document.save(OUTPUT)
    return OUTPUT


if __name__ == "__main__":
    path = build_report()
    print(f"Created DOCX report: {path}")
