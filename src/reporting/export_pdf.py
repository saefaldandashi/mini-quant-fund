"""
PDF Exporter - Converts HTML reports to PDF.

Uses Playwright (preferred) or WeasyPrint as fallback.
"""

import logging
from pathlib import Path
from typing import Optional
import base64

# Try Playwright first (most reliable)
PLAYWRIGHT_AVAILABLE = False
try:
    from playwright.sync_api import sync_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    pass

# Try WeasyPrint as fallback
WEASYPRINT_AVAILABLE = False
try:
    from weasyprint import HTML
    WEASYPRINT_AVAILABLE = True
except (ImportError, OSError):
    pass


class PDFExporter:
    """Exports HTML reports to PDF."""
    
    def __init__(self):
        """Initialize PDF exporter."""
        if PLAYWRIGHT_AVAILABLE:
            logging.info("PDF export: Using Playwright")
        elif WEASYPRINT_AVAILABLE:
            logging.info("PDF export: Using WeasyPrint")
        else:
            logging.warning("PDF export: No PDF library available, will save as HTML only")
    
    def export_html_to_pdf(
        self,
        html_content: str,
        output_pdf_path: Path,
        output_html_path: Optional[Path] = None,
    ) -> Path:
        """
        Export HTML content to PDF.
        
        Args:
            html_content: HTML string
            output_pdf_path: Path for PDF output
            output_html_path: Optional path to save HTML (for debugging)
        
        Returns:
            Path to generated PDF (or HTML if PDF generation fails)
        """
        output_pdf_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save HTML first (always useful)
        if output_html_path is None:
            output_html_path = output_pdf_path.with_suffix('.html')
        
        with open(output_html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        # Try PDF generation
        if PLAYWRIGHT_AVAILABLE:
            try:
                return self._export_with_playwright(html_content, output_pdf_path)
            except Exception as e:
                logging.warning(f"Playwright PDF generation failed: {e}, using HTML")
                return output_html_path
        
        elif WEASYPRINT_AVAILABLE:
            try:
                return self._export_with_weasyprint(html_content, output_pdf_path)
            except Exception as e:
                logging.warning(f"WeasyPrint PDF generation failed: {e}, using HTML")
                return output_html_path
        
        else:
            logging.warning("No PDF library available, saved as HTML only")
            return output_html_path
    
    def _export_with_playwright(
        self,
        html_content: str,
        output_path: Path,
    ) -> Path:
        """Export using Playwright (most reliable)."""
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            
            # Load HTML content
            page.set_content(html_content, wait_until='networkidle')
            
            # Generate PDF
            page.pdf(
                path=str(output_path),
                format='A4',
                print_background=True,
                margin={'top': '2cm', 'right': '2cm', 'bottom': '2cm', 'left': '2cm'},
            )
            
            browser.close()
        
        logging.info(f"PDF generated with Playwright: {output_path}")
        return output_path
    
    def _export_with_weasyprint(
        self,
        html_content: str,
        output_path: Path,
    ) -> Path:
        """Export using WeasyPrint (fallback)."""
        HTML(string=html_content).write_pdf(output_path)
        logging.info(f"PDF generated with WeasyPrint: {output_path}")
        return output_path
    
    def embed_charts_base64(self, chart_paths: dict) -> dict:
        """
        Convert chart images to base64 for embedding in HTML.
        
        Args:
            chart_paths: Dict of chart_name -> file_path
        
        Returns:
            Dict of chart_name -> base64 data URI
        """
        embedded = {}
        
        for name, path in chart_paths.items():
            try:
                chart_path = Path(path)
                if chart_path.exists():
                    with open(chart_path, 'rb') as f:
                        img_data = f.read()
                        img_base64 = base64.b64encode(img_data).decode('utf-8')
                        # Determine image type from extension
                        ext = chart_path.suffix.lower()
                        mime_type = 'image/png' if ext == '.png' else 'image/jpeg'
                        embedded[name] = f"data:{mime_type};base64,{img_base64}"
            except Exception as e:
                logging.warning(f"Could not embed chart {name}: {e}")
        
        return embedded
