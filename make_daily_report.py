from datetime import datetime
from html import escape
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


REPORT_DIR = Path("daily_reports")
W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def paragraph(text, style=None, bullet=False):
    text = escape(text)
    ppr = ""
    if style:
        ppr += f'<w:pStyle w:val="{style}"/>'
    if bullet:
        ppr += '<w:numPr><w:ilvl w:val="0"/><w:numId w:val="1"/></w:numPr>'
    return (
        "<w:p>"
        f"<w:pPr>{ppr}</w:pPr>"
        '<w:r><w:rPr><w:rFonts w:ascii="Microsoft YaHei" w:hAnsi="Microsoft YaHei" '
        'w:eastAsia="Microsoft YaHei"/></w:rPr>'
        f"<w:t>{text}</w:t></w:r></w:p>"
    )


def build_report_body(date):
    sections = [
        (
            "一、今日修改内容",
            [
                "围绕最近接近时间 Tc 的表示方式修改 02_train_transformer.py。",
                "尝试使用两星平均轨道周期 T_avg = 0.5 * (TA + TB) 构造 T_ratio = Tc / T_avg。",
                "最终决定训练仍使用稳定的 Tc / 48h，报告中额外统计 T_ratio。",
                "新增时间段辅助分类头，将最近接近时间划分为 0-6h、6-12h、12-24h、24-36h、36-48h。",
            ],
        ),
        (
            "二、今日实验结果",
            [
                "直接训练 T_ratio 的版本分类较好，但时间回归严重退化，T_MAE 约 642 min。",
                "改回 Tc / 48h 训练后，测试集分类准确率约 97.79%，危险召回率约 97.35%，误报率约 1.92%，漏报率约 2.65%。",
                "改回 Tc / 48h 后，D_MAE 约 1.299 km，T_MAE 约 209.14 min，T_ratio_MAE 约 0.738。",
            ],
        ),
        (
            "三、好的结果",
            [
                "危险/安全分类能力明显增强，危险召回率接近 97%，漏报率降到 3% 以下。",
                "T_ratio 可以作为汇报解释指标，用来说明最近接近发生在多少个平均轨道周期之后。",
                "确认了直接使用 T_ratio 作为训练标签不稳定，改回 Tc / 48h 是正确方向。",
            ],
        ),
        (
            "四、不好的结果",
            [
                "时间回归仍然偏大，当前 T_MAE 约 209 min，距离几十分钟级别仍有差距。",
                "D_MAE 约 1.3 km，尚未稳定压到 1 km 以下。",
                "边界危险样本仍可能漏报，例如 Dc 接近 10 km 的样本。",
            ],
        ),
        (
            "五、原因判断",
            [
                "T_ratio 的数值范围受轨道周期影响较大，直接训练会放大时间尺度差异。",
                "Tc 在 48 小时内跨度较大，单一连续回归容易预测到不准确的时间窗口。",
                "部分样本可能存在多次接近，模型容易学到错误的接近窗口。",
            ],
        ),
        (
            "六、下一步建议",
            [
                "运行加入时间段辅助分类头后的新版 02_train_transformer.py。",
                "重点观察 T_MAE 是否下降、时间段准确率是否较高、D_MAE 是否受影响、危险召回率是否保持稳定。",
                "如果时间段辅助有效，可以继续细化时间段；如果无效，再考虑加入少量物理特征或物理精修。",
            ],
        ),
    ]

    body = [paragraph(f"{date} 实验总结报告", "Title")]
    for title, items in sections:
        body.append(paragraph(title, "Heading1"))
        for item in items:
            body.append(paragraph(item, bullet=True))
    return "".join(body)


def write_docx(path, body):
    document = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="{W_NS}" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
<w:body>
{body}
<w:sectPr><w:pgSz w:w="11906" w:h="16838"/><w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440"/></w:sectPr>
</w:body></w:document>'''

    styles = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="{W_NS}">
  <w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/><w:rPr><w:rFonts w:ascii="Microsoft YaHei" w:hAnsi="Microsoft YaHei" w:eastAsia="Microsoft YaHei"/><w:sz w:val="22"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/><w:rPr><w:rFonts w:ascii="Microsoft YaHei" w:hAnsi="Microsoft YaHei" w:eastAsia="Microsoft YaHei"/><w:b/><w:sz w:val="36"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:basedOn w:val="Normal"/><w:rPr><w:rFonts w:ascii="Microsoft YaHei" w:hAnsi="Microsoft YaHei" w:eastAsia="Microsoft YaHei"/><w:b/><w:color w:val="1F4E79"/><w:sz w:val="28"/></w:rPr></w:style>
</w:styles>'''

    numbering = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:numbering xmlns:w="{W_NS}">
  <w:abstractNum w:abstractNumId="0">
    <w:multiLevelType w:val="hybridMultilevel"/>
    <w:lvl w:ilvl="0"><w:start w:val="1"/><w:numFmt w:val="bullet"/><w:lvlText w:val="•"/><w:lvlJc w:val="left"/><w:pPr><w:ind w:left="720" w:hanging="360"/></w:pPr></w:lvl>
  </w:abstractNum>
  <w:num w:numId="1"><w:abstractNumId w:val="0"/></w:num>
</w:numbering>'''

    content_types = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
  <Override PartName="/word/numbering.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.numbering+xml"/>
</Types>'''
    rels = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>'''
    doc_rels = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/numbering" Target="numbering.xml"/>
</Relationships>'''

    with ZipFile(path, "w", ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types.encode("utf-8"))
        zf.writestr("_rels/.rels", rels.encode("utf-8"))
        zf.writestr("word/_rels/document.xml.rels", doc_rels.encode("utf-8"))
        zf.writestr("word/document.xml", document.encode("utf-8"))
        zf.writestr("word/styles.xml", styles.encode("utf-8"))
        zf.writestr("word/numbering.xml", numbering.encode("utf-8"))


def main():
    REPORT_DIR.mkdir(exist_ok=True)
    date = datetime.now().strftime("%Y-%m-%d")
    path = REPORT_DIR / f"{date}_daily_report.docx"
    write_docx(path, build_report_body(date))
    print(path.resolve())


if __name__ == "__main__":
    main()
