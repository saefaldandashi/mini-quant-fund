"""
PDF Export for Daily Digest.

Converts HTML digest to PDF using Playwright for headless browser rendering.
Falls back to weasyprint if Playwright is not available.
"""

import logging
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

# Check for available PDF engines
PLAYWRIGHT_AVAILABLE = False
WEASYPRINT_AVAILABLE = False
WeasyprintHTML = None

try:
    from playwright.async_api import async_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    pass

try:
    from weasyprint import HTML as WeasyprintHTML
    WEASYPRINT_AVAILABLE = True
except (ImportError, OSError):
    # WeasyPrint may fail if system libraries (like libgobject) are missing
    pass


class PDFExporter:
    """Exports HTML to PDF."""
    
    def __init__(
        self,
        engine: str = "auto",  # 'playwright', 'weasyprint', 'auto'
        page_format: str = "A4",
        margin_top: str = "15mm",
        margin_bottom: str = "15mm",
        margin_left: str = "15mm",
        margin_right: str = "15mm",
    ):
        self.engine = engine
        self.page_format = page_format
        self.margin_top = margin_top
        self.margin_bottom = margin_bottom
        self.margin_left = margin_left
        self.margin_right = margin_right
        
        # Select engine
        if engine == "auto":
            if PLAYWRIGHT_AVAILABLE:
                self.engine = "playwright"
            elif WEASYPRINT_AVAILABLE:
                self.engine = "weasyprint"
            else:
                self.engine = "none"
                logger.warning("No PDF engine available. Install playwright or weasyprint.")
    
    def export(
        self,
        html_path: str,
        pdf_path: str,
        title: Optional[str] = None,
        date: Optional[str] = None,
    ) -> bool:
        """
        Export HTML file to PDF.
        
        Args:
            html_path: Path to source HTML file
            pdf_path: Path to output PDF file
            title: Optional title for header
            date: Optional date for header
            
        Returns:
            True if successful, False otherwise
        """
        html_path = Path(html_path)
        pdf_path = Path(pdf_path)
        
        if not html_path.exists():
            logger.error(f"HTML file not found: {html_path}")
            return False
            
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        
        if self.engine == "playwright":
            return self._export_playwright(html_path, pdf_path, title, date)
        elif self.engine == "weasyprint":
            return self._export_weasyprint(html_path, pdf_path)
        else:
            logger.error("No PDF engine available")
            return False
    
    def export_from_string(
        self,
        html_content: str,
        pdf_path: str,
        title: Optional[str] = None,
        date: Optional[str] = None,
    ) -> bool:
        """
        Export HTML string to PDF.
        
        Args:
            html_content: HTML string
            pdf_path: Path to output PDF file
            title: Optional title for header
            date: Optional date for header
            
        Returns:
            True if successful, False otherwise
        """
        pdf_path = Path(pdf_path)
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        
        if self.engine == "playwright":
            return self._export_playwright_string(html_content, pdf_path, title, date)
        elif self.engine == "weasyprint":
            return self._export_weasyprint_string(html_content, pdf_path)
        else:
            logger.error("No PDF engine available")
            return False
    
    def _export_playwright(
        self,
        html_path: Path,
        pdf_path: Path,
        title: Optional[str],
        date: Optional[str],
    ) -> bool:
        """Export using Playwright."""
        try:
            asyncio.run(self._export_playwright_async(html_path, pdf_path, title, date))
            return True
        except Exception as e:
            logger.error(f"Playwright PDF export failed: {e}")
            return False
    
    async def _export_playwright_async(
        self,
        html_path: Path,
        pdf_path: Path,
        title: Optional[str],
        date: Optional[str],
    ):
        """Async Playwright export."""
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()
            
            # Load HTML file
            await page.goto(f"file://{html_path.absolute()}")
            
            # Wait for content to render
            await page.wait_for_load_state('networkidle')
            
            # Generate PDF
            await page.pdf(
                path=str(pdf_path),
                format=self.page_format,
                margin={
                    'top': self.margin_top,
                    'bottom': self.margin_bottom,
                    'left': self.margin_left,
                    'right': self.margin_right,
                },
                display_header_footer=True,
                header_template=self._get_header_template(title, date),
                footer_template=self._get_footer_template(),
                print_background=True,
            )
            
            await browser.close()
            
        logger.info(f"PDF exported to {pdf_path}")
    
    def _export_playwright_string(
        self,
        html_content: str,
        pdf_path: Path,
        title: Optional[str],
        date: Optional[str],
    ) -> bool:
        """Export HTML string using Playwright."""
        try:
            asyncio.run(self._export_playwright_string_async(
                html_content, pdf_path, title, date
            ))
            return True
        except Exception as e:
            logger.error(f"Playwright PDF export failed: {e}")
            return False
    
    async def _export_playwright_string_async(
        self,
        html_content: str,
        pdf_path: Path,
        title: Optional[str],
        date: Optional[str],
    ):
        """Async Playwright export from string."""
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()
            
            # Set HTML content directly
            await page.set_content(html_content, wait_until='networkidle')
            
            # Generate PDF
            await page.pdf(
                path=str(pdf_path),
                format=self.page_format,
                margin={
                    'top': self.margin_top,
                    'bottom': self.margin_bottom,
                    'left': self.margin_left,
                    'right': self.margin_right,
                },
                display_header_footer=True,
                header_template=self._get_header_template(title, date),
                footer_template=self._get_footer_template(),
                print_background=True,
            )
            
            await browser.close()
            
        logger.info(f"PDF exported to {pdf_path}")
    
    def _export_weasyprint(self, html_path: Path, pdf_path: Path) -> bool:
        """Export using WeasyPrint."""
        try:
            html = WeasyprintHTML(filename=str(html_path))
            html.write_pdf(str(pdf_path))
            logger.info(f"PDF exported to {pdf_path}")
            return True
        except Exception as e:
            logger.error(f"WeasyPrint PDF export failed: {e}")
            return False
    
    def _export_weasyprint_string(self, html_content: str, pdf_path: Path) -> bool:
        """Export HTML string using WeasyPrint."""
        try:
            html = WeasyprintHTML(string=html_content)
            html.write_pdf(str(pdf_path))
            logger.info(f"PDF exported to {pdf_path}")
            return True
        except Exception as e:
            logger.error(f"WeasyPrint PDF export failed: {e}")
            return False
    
    def _get_header_template(
        self,
        title: Optional[str],
        date: Optional[str],
    ) -> str:
        """Generate header template for PDF."""
        title = title or "Daily Intelligence Digest"
        date = date or datetime.now().strftime("%Y-%m-%d")
        
        return f'''
        <div style="font-size: 8px; width: 100%; text-align: center; color: #666; padding: 5px 20px;">
            <span>{title}</span>
            <span style="margin-left: 20px;">{date}</span>
        </div>
        '''
    
    def _get_footer_template(self) -> str:
        """Generate footer template for PDF."""
        return '''
        <div style="font-size: 8px; width: 100%; text-align: center; color: #666; padding: 5px 20px;">
            <span>Page <span class="pageNumber"></span> of <span class="totalPages"></span></span>
            <span style="margin-left: 20px;">Confidential</span>
        </div>
        '''


def export_to_pdf(
    html_path: str,
    pdf_path: str,
    title: Optional[str] = None,
    date: Optional[str] = None,
) -> bool:
    """
    Convenience function to export HTML to PDF.
    
    Args:
        html_path: Path to source HTML
        pdf_path: Path to output PDF
        title: Optional title
        date: Optional date
        
    Returns:
        True if successful
    """
    exporter = PDFExporter()
    return exporter.export(html_path, pdf_path, title, date)


def check_pdf_support() -> dict:
    """Check available PDF export support."""
    return {
        'playwright': PLAYWRIGHT_AVAILABLE,
        'weasyprint': WEASYPRINT_AVAILABLE,
        'any_available': PLAYWRIGHT_AVAILABLE or WEASYPRINT_AVAILABLE,
    }
