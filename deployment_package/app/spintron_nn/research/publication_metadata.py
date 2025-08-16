"""
Publication metadata and data structures for academic publications.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Any, Optional


@dataclass
class PublicationMetadata:
    """Metadata for academic publication."""
    
    title: str
    authors: List[str]
    affiliations: List[str]
    abstract: str
    keywords: List[str]
    target_venue: str
    submission_date: str = ""
    
    def __post_init__(self):
        if not self.submission_date:
            self.submission_date = datetime.now().strftime("%Y-%m-%d")


@dataclass
class ExperimentalSection:
    """Structure for experimental methodology section."""
    
    objective: str
    hypothesis: str
    methodology: str
    participants_or_samples: str
    variables: Dict[str, str]
    controls: List[str]
    measurements: List[str]
    data_collection_period: str = ""
    
    def __post_init__(self):
        if not self.data_collection_period:
            self.data_collection_period = datetime.now().strftime("%Y-%m")


@dataclass
class ResultsSection:
    """Structure for results section."""
    
    primary_findings: List[str]
    statistical_analyses: List[Dict[str, Any]]
    figures: List[Dict[str, str]]
    tables: List[Dict[str, Any]]
    effect_sizes: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]
    
    
@dataclass
class DiscussionSection:
    """Structure for discussion section."""
    
    interpretation: str
    limitations: List[str]
    implications: List[str]
    future_work: List[str]
    conclusions: str


@dataclass
class Citation:
    """Academic citation structure."""
    
    authors: List[str]
    title: str
    venue: str
    year: int
    pages: Optional[str] = None
    doi: Optional[str] = None
    url: Optional[str] = None
    
    def format_apa(self) -> str:
        """Format citation in APA style."""
        author_str = ", ".join(self.authors)
        if len(self.authors) > 1:
            author_str = author_str.rsplit(", ", 1)
            author_str = " & ".join(author_str)
            
        citation = f"{author_str} ({self.year}). {self.title}. {self.venue}"
        
        if self.pages:
            citation += f", {self.pages}"
        if self.doi:
            citation += f". https://doi.org/{self.doi}"
            
        return citation + "."
        
    def format_ieee(self) -> str:
        """Format citation in IEEE style."""
        author_str = ", ".join(self.authors)
        citation = f"{author_str}, \"{self.title},\" {self.venue}"
        
        if self.pages:
            citation += f", pp. {self.pages}"
        citation += f", {self.year}"
        
        if self.doi:
            citation += f", doi: {self.doi}"
            
        return citation + "."


@dataclass  
class PublicationStructure:
    """Complete publication structure."""
    
    metadata: PublicationMetadata
    experimental: ExperimentalSection
    results: ResultsSection
    discussion: DiscussionSection
    citations: List[Citation]
    appendices: List[Dict[str, Any]]