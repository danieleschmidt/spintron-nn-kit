"""
Publication generators for academic papers and reports.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

from .publication_metadata import (
    PublicationMetadata, ExperimentalSection, ResultsSection, 
    DiscussionSection, Citation, PublicationStructure
)


class LaTeXGenerator:
    """Generate LaTeX documents for academic publications."""
    
    def __init__(self, template_style: str = "ieee"):
        self.template_style = template_style
        self.template_map = {
            "ieee": self._ieee_template,
            "acm": self._acm_template,
            "springer": self._springer_template,
            "nature": self._nature_template
        }
        
    def generate_paper(self, structure: PublicationStructure) -> str:
        """Generate complete LaTeX paper."""
        template_func = self.template_map.get(self.template_style, self._ieee_template)
        return template_func(structure)
        
    def _ieee_template(self, structure: PublicationStructure) -> str:
        """Generate IEEE-style LaTeX paper."""
        latex = r"""\documentclass[conference]{IEEEtran}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{algorithmic}
\usepackage{graphicx}
\usepackage{textcomp}
\usepackage{xcolor}

\begin{document}

\title{""" + structure.metadata.title + r"""}

\author{
"""
        
        # Add authors
        for i, author in enumerate(structure.metadata.authors):
            latex += f"\\IEEEauthorblockN{{{author}}}\n"
            if i < len(structure.metadata.affiliations):
                latex += f"\\IEEEauthorblockA{{{structure.metadata.affiliations[i]}}}\n"
                
        latex += r"""
}

\maketitle

\begin{abstract}
""" + structure.metadata.abstract + r"""
\end{abstract}

\begin{IEEEkeywords}
""" + ", ".join(structure.metadata.keywords) + r"""
\end{IEEEkeywords}

\section{Introduction}
% Introduction content will be generated based on experimental objective
""" + structure.experimental.objective + r"""

\section{Methodology}
""" + structure.experimental.methodology + r"""

\section{Results}
"""
        
        # Add primary findings
        for finding in structure.results.primary_findings:
            latex += f"{finding}\n\n"
            
        latex += r"""
\section{Discussion}
""" + structure.discussion.interpretation + r"""

\subsection{Limitations}
"""
        
        for limitation in structure.discussion.limitations:
            latex += f"\\item {limitation}\n"
            
        latex += r"""
\section{Conclusion}
""" + structure.discussion.conclusions + r"""

\begin{thebibliography}{""" + str(len(structure.citations)) + r"""}
"""
        
        # Add citations
        for i, citation in enumerate(structure.citations, 1):
            latex += f"\\bibitem{{ref{i}}} {citation.format_ieee()}\n"
            
        latex += r"""
\end{thebibliography}

\end{document}"""
        
        return latex
        
    def _acm_template(self, structure: PublicationStructure) -> str:
        """Generate ACM-style LaTeX paper."""
        # Simplified ACM template
        return self._ieee_template(structure).replace("IEEEtran", "acmart")
        
    def _springer_template(self, structure: PublicationStructure) -> str:
        """Generate Springer-style LaTeX paper."""
        # Simplified Springer template
        return self._ieee_template(structure).replace("IEEEtran", "llncs")
        
    def _nature_template(self, structure: PublicationStructure) -> str:
        """Generate Nature-style LaTeX paper."""
        # Simplified Nature template
        return self._ieee_template(structure).replace("conference", "article")


class MarkdownGenerator:
    """Generate Markdown documents for publications."""
    
    def generate_paper(self, structure: PublicationStructure) -> str:
        """Generate complete Markdown paper."""
        markdown = f"""# {structure.metadata.title}

## Authors
{', '.join(structure.metadata.authors)}

## Affiliations
{', '.join(structure.metadata.affiliations)}

## Abstract
{structure.metadata.abstract}

## Keywords
{', '.join(structure.metadata.keywords)}

## Introduction
{structure.experimental.objective}

## Methodology
{structure.experimental.methodology}

### Hypothesis
{structure.experimental.hypothesis}

## Results
"""
        
        for finding in structure.results.primary_findings:
            markdown += f"- {finding}\n"
            
        markdown += f"""
## Discussion
{structure.discussion.interpretation}

### Limitations
"""
        
        for limitation in structure.discussion.limitations:
            markdown += f"- {limitation}\n"
            
        markdown += f"""
### Future Work
"""
        
        for future in structure.discussion.future_work:
            markdown += f"- {future}\n"
            
        markdown += f"""
## Conclusion
{structure.discussion.conclusions}

## References
"""
        
        for i, citation in enumerate(structure.citations, 1):
            markdown += f"{i}. {citation.format_apa()}\n"
            
        return markdown


class HTMLGenerator:
    """Generate HTML documents for publications."""
    
    def generate_paper(self, structure: PublicationStructure) -> str:
        """Generate complete HTML paper."""
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{structure.metadata.title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }}
        .abstract {{ background-color: #f5f5f5; padding: 15px; border-left: 4px solid #333; }}
        .keywords {{ font-style: italic; color: #666; }}
        .section {{ margin: 20px 0; }}
        .authors {{ color: #444; }}
        .citations {{ font-size: 0.9em; }}
    </style>
</head>
<body>
    <h1>{structure.metadata.title}</h1>
    
    <div class="authors">
        <strong>Authors:</strong> {', '.join(structure.metadata.authors)}
    </div>
    
    <div class="authors">
        <strong>Affiliations:</strong> {', '.join(structure.metadata.affiliations)}
    </div>
    
    <div class="abstract">
        <h3>Abstract</h3>
        <p>{structure.metadata.abstract}</p>
    </div>
    
    <div class="keywords">
        <strong>Keywords:</strong> {', '.join(structure.metadata.keywords)}
    </div>
    
    <div class="section">
        <h2>Introduction</h2>
        <p>{structure.experimental.objective}</p>
    </div>
    
    <div class="section">
        <h2>Methodology</h2>
        <p>{structure.experimental.methodology}</p>
        <h3>Hypothesis</h3>
        <p>{structure.experimental.hypothesis}</p>
    </div>
    
    <div class="section">
        <h2>Results</h2>
        <ul>
"""
        
        for finding in structure.results.primary_findings:
            html += f"            <li>{finding}</li>\n"
            
        html += f"""        </ul>
    </div>
    
    <div class="section">
        <h2>Discussion</h2>
        <p>{structure.discussion.interpretation}</p>
        
        <h3>Limitations</h3>
        <ul>
"""
        
        for limitation in structure.discussion.limitations:
            html += f"            <li>{limitation}</li>\n"
            
        html += f"""        </ul>
        
        <h3>Future Work</h3>
        <ul>
"""
        
        for future in structure.discussion.future_work:
            html += f"            <li>{future}</li>\n"
            
        html += f"""        </ul>
    </div>
    
    <div class="section">
        <h2>Conclusion</h2>
        <p>{structure.discussion.conclusions}</p>
    </div>
    
    <div class="section citations">
        <h2>References</h2>
        <ol>
"""
        
        for citation in structure.citations:
            html += f"            <li>{citation.format_apa()}</li>\n"
            
        html += """        </ol>
    </div>
</body>
</html>"""
        
        return html


class PublicationExporter:
    """Export publications in multiple formats."""
    
    def __init__(self):
        self.latex_generator = LaTeXGenerator()
        self.markdown_generator = MarkdownGenerator()
        self.html_generator = HTMLGenerator()
        
    def export_all_formats(self, structure: PublicationStructure, 
                          output_dir: str = "publication_output") -> Dict[str, str]:
        """Export publication in all supported formats."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Generate safe filename
        safe_title = "".join(c for c in structure.metadata.title if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_title = safe_title.replace(' ', '_')
        
        files_created = {}
        
        # LaTeX
        latex_content = self.latex_generator.generate_paper(structure)
        latex_file = output_path / f"{safe_title}.tex"
        latex_file.write_text(latex_content, encoding='utf-8')
        files_created['latex'] = str(latex_file)
        
        # Markdown
        markdown_content = self.markdown_generator.generate_paper(structure)
        markdown_file = output_path / f"{safe_title}.md"
        markdown_file.write_text(markdown_content, encoding='utf-8')
        files_created['markdown'] = str(markdown_file)
        
        # HTML
        html_content = self.html_generator.generate_paper(structure)
        html_file = output_path / f"{safe_title}.html"
        html_file.write_text(html_content, encoding='utf-8')
        files_created['html'] = str(html_file)
        
        # JSON metadata
        metadata_file = output_path / f"{safe_title}_metadata.json"
        metadata_file.write_text(json.dumps(structure.metadata.__dict__, indent=2), encoding='utf-8')
        files_created['metadata'] = str(metadata_file)
        
        return files_created