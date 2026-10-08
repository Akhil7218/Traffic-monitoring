"""
TrafficSentinel AI — Automated PDF Citation & Fine Schedule Builder
Generates official e-challan PDFs using ReportLab with embedded evidence frames and QR code payment links.
"""

import os
import qrcode
import sqlite3
import tempfile
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                 Table, TableStyle, Image as RLImage)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

import config

# Statutory Penalty Schedule (Motor Vehicles Act 1988 / 2019 Amendments)
BASE_FINES = {
    "NO HELMET":     1000,
    "TRIPLE RIDING": 1000,
    "WRONG WAY":     5000,
    "OVERSPEEDING":  2000,
}

SEVERITY_MULTIPLIER = {
    1: 1.0,   # First Infraction
    2: 2.0,   # Second Infraction
    3: 3.0,   # Repeat / Habitual Infraction
}

ONLINE_PAYMENT_PORTAL = "https://parivahan.gov.in/challan/"


def query_prior_offences(db_path, plate_number):
    """Counts recorded infractions for the specified plate."""
    if not plate_number or plate_number == "UNKNOWN":
        return 0
    try:
        connection = sqlite3.connect(db_path)
        cursor = connection.cursor()
        cursor.execute("SELECT COUNT(*) FROM violations WHERE plate=?", (plate_number,))
        count = cursor.fetchone()[0]
        connection.close()
        return count
    except Exception:
        return 0


def compute_infraction_fee(violations_list, offence_count):
    """Calculates total fine amount based on violations and severity multiplier."""
    base_sum = sum(BASE_FINES.get(v.strip(), 500) for v in violations_list)
    multiplier = SEVERITY_MULTIPLIER.get(min(offence_count, 3), 3.0)
    total_fee = int(base_sum * multiplier)
    return base_sum, multiplier, total_fee


def generate_payment_qr(citation_id, total_fee):
    """Creates a temporary PNG QR code containing the payment URL."""
    portal_url = f"{ONLINE_PAYMENT_PORTAL}TS{citation_id:06d}?amount={total_fee}"
    qr_object = qrcode.QRCode(version=1, box_size=6, border=2)
    qr_object.add_data(portal_url)
    qr_object.make(fit=True)
    qr_image = qr_object.make_image(fill_color="black", back_color="white")
    temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    qr_image.save(temp_file.name)
    return temp_file.name, portal_url


def build_citation_pdf(challan_dir, screenshot_dir,
                       violation_id, timestamp, video_source,
                       violation_str, plate_number, screenshot_filename,
                       db_path, owner_name="Not Available",
                       offence_count=None):
    """
    Builds a professional PDF citation document with embedded evidence.
    """
    pdf_filename = f"challan_TM{violation_id:06d}.pdf"
    pdf_filepath = os.path.join(challan_dir, pdf_filename)

    violations_list = [v.strip() for v in violation_str.split("+")]
    if offence_count is None:
        offence_count = query_prior_offences(db_path, plate_number)
    
    base_fee, multiplier, total_fee = compute_infraction_fee(violations_list, offence_count)

    is_repeat = offence_count > 1
    severity_label = "HABITUAL OFFENDER" if offence_count >= 3 else ("REPEAT OFFENDER" if is_repeat else "FIRST OFFENCE")
    badge_color = colors.HexColor('#c0392b') if offence_count >= 3 else (
                  colors.HexColor('#e67e22') if is_repeat else colors.HexColor('#27ae60'))

    qr_path, payment_url = generate_payment_qr(violation_id, total_fee)

    doc = SimpleDocTemplate(
        pdf_filepath, pagesize=A4,
        rightMargin=1.8 * cm, leftMargin=1.8 * cm,
        topMargin=1.5 * cm, bottomMargin=1.5 * cm
    )

    # Document Styles
    hdr_style = ParagraphStyle('hdr', fontSize=18, fontName='Helvetica-Bold',
                               alignment=TA_CENTER, textColor=colors.HexColor('#c0392b'),
                               spaceAfter=6, spaceBefore=4)
    sub_hdr_style = ParagraphStyle('sub_hdr', fontSize=9, fontName='Helvetica',
                                   alignment=TA_CENTER, textColor=colors.HexColor('#555555'),
                                   spaceAfter=3, leading=14)
    section_style = ParagraphStyle('sec', fontSize=11, fontName='Helvetica-Bold',
                                   textColor=colors.HexColor('#2c3e50'), spaceBefore=10, spaceAfter=4)
    foot_style = ParagraphStyle('foot', fontSize=7.5, fontName='Helvetica',
                                alignment=TA_CENTER, textColor=colors.grey)

    story = []

    # Title & Subtitle Header
    story.append(Paragraph("TRAFFIC SENTINEL AI — OFFICIAL CITATION", hdr_style))
    story.append(Spacer(1, 0.15 * cm))
    story.append(Paragraph("Automated Smart Traffic Enforcement System  |  Vision AI Surveillance", sub_hdr_style))
    story.append(Paragraph("Issued under Motor Vehicles Act, 1988 (Amended 2019)", sub_hdr_style))
    story.append(Spacer(1, 0.35 * cm))

    # Citation Banner
    banner_text = f"CITATION NO: TS-{violation_id:06d}   |   {severity_label}   |   RECORDED INFRACTION #{offence_count}"
    banner_table = Table([[Paragraph(banner_text, ParagraphStyle('b_text', fontSize=10, fontName='Helvetica-Bold', textColor=colors.white, alignment=TA_CENTER))]], colWidths=[17.4 * cm])
    banner_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), badge_color),
        ('TOPPADDING', (0, 0), (-1, -1), 7),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 7),
    ]))
    story.append(banner_table)
    story.append(Spacer(1, 0.4 * cm))

    # Vehicle & Registration Details
    story.append(Paragraph("Vehicle & Registration Record", section_style))
    vehicle_details = [
        ["License Plate", plate_number if plate_number != "UNKNOWN" else "Unidentified", "Owner Name", owner_name],
        ["Date & Timestamp", timestamp, "Video Source", video_source[:35] + "..." if len(video_source) > 35 else video_source],
        ["Previous Infractions", str(offence_count - 1), "Offender Category", severity_label],
    ]
    detail_table = Table(vehicle_details, colWidths=[4 * cm, 4.7 * cm, 4 * cm, 4.7 * cm])
    detail_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (2, 0), (2, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#dfe6e9')),
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f0f4f8')),
        ('BACKGROUND', (2, 0), (2, -1), colors.HexColor('#f0f4f8')),
        ('ROWBACKGROUNDS', (0, 0), (-1, -1), [colors.white, colors.HexColor('#fafafa')]),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('TEXTCOLOR', (1, 2), (1, 2), badge_color),
        ('FONTNAME', (1, 2), (1, 2), 'Helvetica-Bold'),
        ('TEXTCOLOR', (3, 2), (3, 2), badge_color),
        ('FONTNAME', (3, 2), (3, 2), 'Helvetica-Bold'),
    ]))
    story.append(detail_table)
    story.append(Spacer(1, 0.3 * cm))

    # Infraction Breakdown Table
    story.append(Paragraph("Infraction Specification & Fee Breakdown", section_style))
    statutory_sections = {
        "NO HELMET":     "Sec 129 MV Act",
        "TRIPLE RIDING": "Sec 128 MV Act",
        "WRONG WAY":     "Sec 184 MV Act",
        "OVERSPEEDING":  "Sec 183 MV Act",
    }
    fine_breakdown_rows = [["#", "Infraction", "Statutory Clause", "Base Penalty"]]
    for idx, infraction_item in enumerate(violations_list, 1):
        clean_item = infraction_item.strip()
        fine_breakdown_rows.append([
            str(idx), clean_item,
            statutory_sections.get(clean_item, "MV Act Provisions"),
            f"Rs. {BASE_FINES.get(clean_item, 500):,}"
        ])
    fine_breakdown_rows.append(["", "Base Total", "", f"Rs. {base_fee:,}"])
    fine_breakdown_rows.append(["", f"Multiplier ({multiplier}x — Incident #{offence_count})", "", ""])
    fine_breakdown_rows.append(["", "TOTAL PAYABLE FINE", "", f"Rs. {total_fee:,}"])

    breakdown_table = Table(fine_breakdown_rows, colWidths=[1 * cm, 7 * cm, 5 * cm, 4.4 * cm])
    breakdown_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c3e50')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -2), 0.5, colors.HexColor('#dfe6e9')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -4), [colors.white, colors.HexColor('#fafafa')]),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('BACKGROUND', (0, -3), (-1, -3), colors.HexColor('#eaf0fb')),
        ('FONTNAME', (0, -3), (-1, -3), 'Helvetica-Bold'),
        ('BACKGROUND', (0, -2), (-1, -2), colors.HexColor('#fff3cd')),
        ('TEXTCOLOR', (0, -2), (-1, -2), colors.HexColor('#856404')),
        ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor('#c0392b')),
        ('TEXTCOLOR', (0, -1), (-1, -1), colors.white),
        ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, -1), (-1, -1), 11),
    ]))
    story.append(breakdown_table)
    story.append(Spacer(1, 0.4 * cm))

    # Image Evidence & Payment QR
    story.append(Paragraph("Visual Evidence & E-Payment", section_style))
    ss_filepath = os.path.join(screenshot_dir, screenshot_filename) if screenshot_filename else None
    evidence_column = []

    if ss_filepath and os.path.exists(ss_filepath):
        evidence_column.append(RLImage(ss_filepath, width=10.5 * cm, height=6 * cm))
    else:
        evidence_column.append(Paragraph("Evidence Frame Unavailable", ParagraphStyle('ns', fontSize=9)))

    qr_image_element = RLImage(qr_path, width=4.5 * cm, height=4.5 * cm)
    payment_descriptor = Paragraph(
        f"<b>Scan to Pay Fine</b><br/><br/>"
        f"Amount: <b>Rs. {total_fee:,}</b><br/><br/>"
        f"Ref: TS-{violation_id:06d}<br/><br/>"
        f"Grace Period: 60 Days<br/><br/>"
        f"<font size='7'>{payment_url[:40]}</font>",
        ParagraphStyle('pay_desc', fontSize=9, fontName='Helvetica', alignment=TA_CENTER, leading=14)
    )
    qr_table = Table([[qr_image_element], [payment_descriptor]], colWidths=[5.9 * cm])
    qr_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))

    evidence_table = Table([[evidence_column, qr_table]], colWidths=[11 * cm, 6.4 * cm])
    evidence_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#dfe6e9')),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(evidence_table)
    story.append(Spacer(1, 0.3 * cm))

    # Payment Notice Block
    instructions_block = [
        [Paragraph("<b>SETTLEMENT INSTRUCTIONS</b>", ParagraphStyle('inst_h', fontSize=10, fontName='Helvetica-Bold', textColor=colors.white, alignment=TA_CENTER))],
        [Paragraph(
            "1. Settle online via <b>parivahan.gov.in</b> or using the direct QR payment portal link above.<br/>"
            "2. Physical payment accepted at authorized <b>Traffic Enforcement Depots</b> or <b>e-Seva Outlets</b>.<br/>"
            "3. Failure to settle within 60 days initiates automated judicial proceedings and license suspension.<br/>"
            "4. Support & Appeals Helpline: <b>1800-SENTINEL</b> | contact@trafficsentinel.ai",
            ParagraphStyle('inst_b', fontSize=8.5, fontName='Helvetica', leading=14)
        )],
    ]
    instructions_table = Table(instructions_block, colWidths=[17.4 * cm])
    instructions_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#27ae60')),
        ('BACKGROUND', (0, 1), (-1, 1), colors.HexColor('#eafaf1')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#a9dfbf')),
        ('TOPPADDING', (0, 0), (-1, -1), 7),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 7),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
    ]))
    story.append(instructions_table)
    story.append(Spacer(1, 0.2 * cm))

    # Official Footnote
    story.append(Paragraph(
        f"TrafficSentinel AI — System Generated Legal Citation. "
        f"Issued: {datetime.now().strftime('%d %b %Y at %H:%M:%S')}  |  "
        f"Ref Code: TS-{violation_id:06d}",
        foot_style
    ))

    doc.build(story)

    try:
        os.unlink(qr_path)
    except Exception:
        pass

    return pdf_filename


# Backward Compatibility Aliases
generate_challan = build_citation_pdf
calculate_fine = compute_infraction_fee
get_offence_count = query_prior_offences
generate_qr = generate_payment_qr
