import asyncio
from pathlib import Path
from qbr.ports.pdf_extractor import PDFPage, PDFTable, PDFTextBlock, PDFSource
from qbr.adapters.pdf.pdf_processing_adapter import PDFProcessingAdapter, ProcessingOptions
from qbr.adapters.pdf.pymupdf_adapter import PyMuPDFAdapter

def test_table_batch_preserves_row_ids_and_headers():
    table = PDFTable((('KPI','Old','New'),('MAU','640','875'),('Adoption','61%','74%')),1,'p1-t0')
    fragments = PDFProcessingAdapter(ProcessingOptions(max_table_rows_per_fragment=2)).process_page('d',PDFPage(1,'', (table,)))
    assert len(fragments)==2
    assert fragments[1].tables[0].row_offset==2
    assert fragments[1].tables[0].header_rows==(('KPI','Old','New'),)
    assert fragments[1].tables[0].cell_ids[0][0]=='p1-t0-r2-c0'

def test_text_inside_table_is_removed_when_block_geometry_known():
    table=PDFTable((('MAU','640','875'),),1,'p1-t0',(0,0,200,100))
    blocks=(PDFTextBlock('p1-b0','MAU 640 875',(10,10,100,50)), PDFTextBlock('p1-b1','Executive summary',(0,150,200,200)))
    result=PDFProcessingAdapter().process_page('d',PDFPage(1,'MAU 640 875\nExecutive summary',(table,),blocks=blocks))
    assert [f.text for f in result if f.text]==['Executive summary']
    assert sum(bool(f.tables) for f in result)==1

def test_normalize_preserves_numeric_signs():
    assert PDFProcessingAdapter.normalize_text(' -12.5%\t +3 ')=='-12.5% +3'

def test_real_pdf_all_pages():
    source=Path('/mnt/data/01_Northwind_Technologies_Q2_2026.pdf')
    if not source.exists(): return
    async def collect():
        return [p async for p in PyMuPDFAdapter().iter_pages(PDFSource('northwind',source))]
    pages=asyncio.run(collect())
    import pymupdf
    with pymupdf.open(source) as doc: assert len(pages)==doc.page_count
    assert pages[0].has_content
    assert any(t.cells and t.table_id for p in pages for t in p.tables)

def test_section_headings_attached_to_content_not_emitted_alone():
    blocks = (
        PDFTextBlock('b0', 'Executive Overview', (0, 0, 100, 10)),
        PDFTextBlock('b1', 'Customer health is positive.', (0, 12, 200, 30)),
        PDFTextBlock('b2', 'Risks', (0, 40, 100, 50)),
        PDFTextBlock('b3', 'Low adoption could delay rollout.', (0, 55, 200, 70)),
    )
    fragments = PDFProcessingAdapter().process_page('d', PDFPage(1, '', blocks=blocks))
    assert [(f.section, f.text) for f in fragments] == [
        ('executive_overview', 'Customer health is positive.'),
        ('risks', 'Low adoption could delay rollout.'),
    ]
    assert all(f.source_text == '' for f in fragments)


def test_table_context_and_source_ids():
    blocks = (PDFTextBlock('b0', 'Performance', (0, 0, 100, 10)),)
    table = PDFTable((('Metric', 'Q1', 'Q2'), ('MAU', '640', '875')), 1, 'p1-t0', (0, 20, 200, 80))
    fragments = PDFProcessingAdapter().process_page('d', PDFPage(1, '', (table,), blocks=blocks))
    assert len(fragments) == 1
    assert fragments[0].section == 'performance'
    assert fragments[0].content_type == 'table'
    assert fragments[0].source_ids == ('p1-t0',)


def test_source_text_opt_in_for_audit_compatibility():
    page = PDFPage(1, 'Executive Overview\n\nCustomer health improved.')
    fragments = PDFProcessingAdapter(ProcessingOptions(include_source_text=True)).process_page('d', page)
    assert fragments[0].source_text == page.text
