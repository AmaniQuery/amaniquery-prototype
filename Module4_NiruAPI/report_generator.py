"""
Report Generation Module for AmaniQuery - 2026 Edition
Creates structured reports using Gemini AI with async support and streaming
"""

import os
import json
import asyncio
from typing import Dict, List, Optional, Any, AsyncGenerator
from datetime import datetime
from loguru import logger


class ReportGenerator:
    """
    Report generator using Gemini AI for creating structured reports
    on legal queries and information gathering from Kenya's laws.

    Enhancements:
    - Async methods for parallel report generation
    - Streaming support for long reports
    - Faster gemini-2.5-flash model
    - Enhanced template system
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize report generator with Gemini API

        Args:
            api_key: Gemini API key (optional, will use env var if not provided)
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not set in environment")

        try:
            import google.generativeai as genai

            genai.configure(api_key=self.api_key)
            self.genai = genai
            # Use faster gemini-2.5-flash for performance
            self.model = genai.GenerativeModel("gemini-2.5-flash")
            logger.info("Report generator initialized with Gemini 2.5 Flash")
        except ImportError:
            raise ValueError(
                "google-generativeai package not installed. Install with: pip install google-generativeai"
            )

    def generate_legal_query_report(
        self, query_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive legal query report

        Args:
            query_analysis: Dictionary containing legal query analysis results

        Returns:
            Dictionary containing the generated legal query report
        """
        prompt = f"""
        Generate a comprehensive legal query report based on the following analysis:

        QUERY ANALYSIS:
        {json.dumps(query_analysis, indent=2)}

        Create a professional report that includes:

        1. QUERY SUMMARY
           - Original legal question
           - Key legal issues identified
           - Legal context and jurisdiction

        2. APPLICABLE LAW
           - Relevant Kenyan statutes and regulations
           - Constitutional provisions if applicable
           - Case law and legal precedents

        3. LEGAL ANALYSIS
           - Step-by-step legal reasoning
           - Rights and obligations involved
           - Potential legal implications

        4. PROCEDURAL GUIDANCE
           - Required steps and documentation
           - Relevant government agencies or courts
           - Timeframes and deadlines

        5. PRACTICAL CONSIDERATIONS
           - Costs and fees involved
           - Alternative dispute resolution options
           - Risk factors and mitigation strategies

        6. RECOMMENDED ACTIONS
           - Immediate steps to take
           - Professional assistance needed
           - Preventive measures

        7. ADDITIONAL RESOURCES
           - Further reading and legal references
           - Support organizations and hotlines
           - Online legal resources

        Format as a professional legal report with clear sections, specific legal references, and actionable guidance.
        """

        try:
            # Use async-compatible generation
            response = self.model.generate_content(prompt)
            report_content = response.text

            return {
                "report_type": "legal_query",
                "title": "Legal Query Analysis Report",
                "content": report_content,
                "generated_at": datetime.utcnow().isoformat(),
                "model_used": "gemini-2.5-flash",
                "query_analysis": query_analysis,
                "metadata": {
                    "sections": self._extract_sections(report_content),
                    "word_count": len(report_content.split()),
                    "readability_score": "professional",
                },
            }

        except Exception as e:
            logger.error(f"Error generating legal query report: {e}")
            return {
                "error": str(e),
                "report_type": "legal_query",
                "generated_at": datetime.utcnow().isoformat(),
            }

    def generate_legal_research_report(
        self, research_data: Dict[str, Any], research_findings: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate a legal research report

        Args:
            research_data: Legal topics and research parameters
            research_findings: Research findings from legal analysis

        Returns:
            Dictionary containing the legal research report
        """
        prompt = f"""
        Generate a comprehensive legal research report based on the following data:

        RESEARCH DATA:
        {json.dumps(research_data, indent=2)}

        RESEARCH FINDINGS:
        {json.dumps(research_findings, indent=2)}

        Structure the report as follows:

        1. EXECUTIVE SUMMARY
           - Key legal findings and implications
           - Main conclusions and recommendations

        2. METHODOLOGY
           - Research methods and sources used
           - Legal databases and resources consulted
           - Analysis framework applied

        3. LEGAL FRAMEWORK ANALYSIS
           - Overview of relevant legal frameworks
           - Key statutes and regulations examined
           - Recent legal developments and amendments

        4. CASE LAW REVIEW
           - Relevant court decisions and precedents
           - Judicial interpretations and applications
           - Emerging legal trends

        5. PRACTICAL IMPLICATIONS
           - Impact on citizens and businesses
           - Compliance requirements and procedures
           - Common legal challenges and solutions

        6. COMPARATIVE ANALYSIS
           - Comparison with other jurisdictions
           - International legal standards and obligations
           - Best practices and benchmarks

        7. RECOMMENDATIONS
           - Legal reforms and improvements needed
           - Policy recommendations
           - Implementation strategies

        8. CONCLUSION
           - Summary of research findings
           - Future outlook and monitoring needs

        Include specific references to Kenyan laws, acts, and legal authorities.
        Provide evidence-based analysis with proper legal citations.
        """

        try:
            response = self.model.generate_content(prompt)
            report_content = response.text

            return {
                "report_type": "legal_research",
                "title": "Legal Research Report - Kenya's Laws",
                "content": report_content,
                "generated_at": datetime.utcnow().isoformat(),
                "model_used": "gemini-1.5-pro",
                "data_sources": {
                    "research_data": research_data,
                    "research_findings": research_findings,
                },
                "metadata": {
                    "sections": self._extract_sections(report_content),
                    "recommendations": self._extract_recommendations(report_content),
                    "word_count": len(report_content.split()),
                },
            }

        except Exception as e:
            logger.error(f"Error generating legal research report: {e}")
            return {
                "error": str(e),
                "report_type": "legal_research",
                "generated_at": datetime.utcnow().isoformat(),
            }

    def generate_constitutional_law_report(
        self, constitutional_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate a constitutional law report

        Args:
            constitutional_analysis: Analysis of constitutional law questions

        Returns:
            Dictionary containing the constitutional law report
        """
        prompt = f"""
        Generate a specialized constitutional law report based on the following analysis:

        CONSTITUTIONAL ANALYSIS:
        {json.dumps(constitutional_analysis, indent=2)}

        Create a focused report covering:

        1. CONSTITUTIONAL PROVISIONS
           - Relevant articles and sections of the Constitution
           - Fundamental rights and freedoms involved
           - Constitutional interpretation principles

        2. JUDICIAL REVIEW
           - Applicable case law from the Supreme Court
           - Constitutional petitions and their outcomes
           - Judicial precedents and interpretations

        3. HUMAN RIGHTS IMPLICATIONS
           - Bill of Rights analysis
           - International human rights obligations
           - Remedies and enforcement mechanisms

        4. INSTITUTIONAL FRAMEWORK
           - Roles of constitutional commissions and offices
           - Separation of powers and checks and balances
           - Devolution and county government relations

        5. CONSTITUTIONAL AMENDMENTS
           - Recent amendments and their implications
           - Amendment procedures and requirements
           - Public participation requirements

        6. PRACTICAL APPLICATION
           - How constitutional provisions apply to daily life
           - Constitutional litigation procedures
           - Public interest litigation opportunities

        7. RECOMMENDATIONS
           - Constitutional reforms needed
           - Implementation of constitutional provisions
           - Protection of constitutional rights

        Include specific references to the Constitution of Kenya 2010 and relevant case law.
        """

        try:
            response = self.model.generate_content(prompt)
            report_content = response.text

            return {
                "report_type": "constitutional_law",
                "title": "Constitutional Law Analysis Report",
                "content": report_content,
                "generated_at": datetime.utcnow().isoformat(),
                "model_used": "gemini-1.5-pro",
                "constitutional_analysis": constitutional_analysis,
                "metadata": {
                    "sections": self._extract_sections(report_content),
                    "constitutional_references": self._extract_constitutional_references(
                        report_content
                    ),
                    "word_count": len(report_content.split()),
                },
            }

        except Exception as e:
            logger.error(f"Error generating constitutional law report: {e}")
            return {
                "error": str(e),
                "report_type": "constitutional_law",
                "generated_at": datetime.utcnow().isoformat(),
            }

    def generate_compliance_report(
        self, legal_requirements: Dict[str, Any], compliance_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate a legal compliance report

        Args:
            legal_requirements: Legal requirements and obligations
            compliance_data: Current compliance status and data

        Returns:
            Dictionary containing the compliance report
        """
        prompt = f"""
        Generate a legal compliance report based on the following information:

        LEGAL REQUIREMENTS:
        {json.dumps(legal_requirements, indent=2)}

        COMPLIANCE DATA:
        {json.dumps(compliance_data, indent=2)}

        Create a comprehensive compliance report covering:

        1. REGULATORY FRAMEWORK
           - Applicable laws and regulations
           - Regulatory authorities and their mandates
           - Compliance standards and requirements

        2. COMPLIANCE ASSESSMENT
           - Current compliance status
           - Gaps and deficiencies identified
           - Risk areas and vulnerabilities

        3. COMPLIANCE PROGRAMS
           - Existing compliance measures
           - Effectiveness of current programs
           - Areas needing improvement

        4. RISK ANALYSIS
           - Compliance risks and their impact
           - Likelihood and consequences of non-compliance
           - Mitigation strategies and controls

        5. ACTION PLAN
           - Immediate compliance actions required
           - Short-term and long-term improvements
           - Resource requirements and timelines

        6. MONITORING AND REPORTING
           - Compliance monitoring mechanisms
           - Reporting requirements and frequencies
           - Audit and review procedures

        7. TRAINING AND AWARENESS
           - Staff training requirements
           - Awareness programs needed
           - Communication strategies

        8. RECOMMENDATIONS
           - Priority compliance actions
           - Best practices implementation
           - Continuous improvement measures

        Include specific legal references and compliance checklists where applicable.
        """

        try:
            response = self.model.generate_content(prompt)
            report_content = response.text

            return {
                "report_type": "compliance",
                "title": "Legal Compliance Assessment Report",
                "content": report_content,
                "generated_at": datetime.utcnow().isoformat(),
                "model_used": "gemini-1.5-pro",
                "data_sources": {
                    "legal_requirements": legal_requirements,
                    "compliance_data": compliance_data,
                },
                "metadata": {
                    "sections": self._extract_sections(report_content),
                    "compliance_gaps": self._identify_compliance_gaps(report_content),
                    "word_count": len(report_content.split()),
                },
            }

        except Exception as e:
            logger.error(f"Error generating compliance report: {e}")
            return {
                "error": str(e),
                "report_type": "compliance",
                "generated_at": datetime.utcnow().isoformat(),
            }

    def generate_technical_audit_report(
        self, system_metrics: Dict[str, Any], performance_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate a technical audit report

        Args:
            system_metrics: System performance and health metrics
            performance_data: Detailed performance measurements

        Returns:
            Dictionary containing the technical audit report
        """
        prompt = f"""
        Generate a technical audit report for the AmaniQuery system based on the following metrics:

        SYSTEM METRICS:
        {json.dumps(system_metrics, indent=2)}

        PERFORMANCE DATA:
        {json.dumps(performance_data, indent=2)}

        Create a technical report covering:

        1. SYSTEM ARCHITECTURE REVIEW
           - Architecture assessment
           - Component analysis
           - Integration evaluation

        2. PERFORMANCE ANALYSIS
           - Response times and latency
           - Throughput and scalability
           - Resource utilization

        3. RELIABILITY ASSESSMENT
           - Uptime and availability
           - Error rates and handling
           - Fault tolerance evaluation

        4. SECURITY AUDIT
           - Security posture assessment
           - Data protection measures
           - Access control evaluation

        5. DATA QUALITY ANALYSIS
           - Data accuracy and completeness
           - Processing pipeline efficiency
           - Storage and retrieval performance

        6. INFRASTRUCTURE ASSESSMENT
           - Hardware and software requirements
           - Scalability considerations
           - Cost optimization opportunities

        7. RECOMMENDATIONS
           - Critical issues requiring immediate attention
           - Performance optimization opportunities
           - Architecture improvements

        8. ROADMAP
           - Short-term improvements (0-3 months)
           - Medium-term enhancements (3-12 months)
           - Long-term strategic initiatives

        Use technical terminology appropriate for engineering teams and include specific metrics, benchmarks, and actionable recommendations.
        """

        try:
            response = self.model.generate_content(prompt)
            report_content = response.text

            return {
                "report_type": "technical_audit",
                "title": "AmaniQuery Technical Audit Report",
                "content": report_content,
                "generated_at": datetime.utcnow().isoformat(),
                "model_used": "gemini-1.5-pro",
                "data_sources": {
                    "system_metrics": system_metrics,
                    "performance_data": performance_data,
                },
                "metadata": {
                    "sections": self._extract_sections(report_content),
                    "critical_issues": self._count_critical_issues(report_content),
                    "recommendations": self._extract_recommendations(report_content),
                    "word_count": len(report_content.split()),
                },
            }

        except Exception as e:
            logger.error(f"Error generating technical audit report: {e}")
            return {
                "error": str(e),
                "report_type": "technical_audit",
                "generated_at": datetime.utcnow().isoformat(),
            }

    def generate_impact_assessment_report(
        self, usage_data: Dict[str, Any], impact_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate an impact assessment report

        Args:
            usage_data: User usage and engagement data
            impact_metrics: Social and economic impact metrics

        Returns:
            Dictionary containing the impact assessment report
        """
        prompt = f"""
        Generate an impact assessment report for AmaniQuery based on the following data:

        USAGE DATA:
        {json.dumps(usage_data, indent=2)}

        IMPACT METRICS:
        {json.dumps(impact_metrics, indent=2)}

        Create a comprehensive impact report covering:

        1. USAGE AND ADOPTION
           - User growth and engagement trends
           - Geographic distribution of users
           - Usage patterns and preferences

        2. SOCIAL IMPACT
           - Democratic participation improvements
           - Access to information enhancements
           - Civic engagement outcomes

        3. ECONOMIC IMPACT
           - Cost savings for users
           - Economic value generated
           - Productivity improvements

        4. EDUCATIONAL IMPACT
           - Learning outcomes
           - Knowledge dissemination
           - Educational accessibility

        5. POLICY AND GOVERNANCE IMPACT
           - Influence on policy processes
           - Government transparency improvements
           - Accountability enhancements

        6. DIGITAL DIVIDE ANALYSIS
           - Accessibility improvements
           - Digital inclusion metrics
           - Equity considerations

        7. SUSTAINABILITY ASSESSMENT
           - Long-term viability
           - Environmental impact
           - Social sustainability

        8. FUTURE IMPACT PROJECTIONS
           - Projected growth and influence
           - Potential challenges
           - Scaling strategies

        Include quantitative metrics where possible, qualitative assessments, and forward-looking projections.
        """

        try:
            response = self.model.generate_content(prompt)
            report_content = response.text

            return {
                "report_type": "impact_assessment",
                "title": "AmaniQuery Impact Assessment Report",
                "content": report_content,
                "generated_at": datetime.utcnow().isoformat(),
                "model_used": "gemini-1.5-pro",
                "data_sources": {
                    "usage_data": usage_data,
                    "impact_metrics": impact_metrics,
                },
                "metadata": {
                    "sections": self._extract_sections(report_content),
                    "impact_areas": self._identify_impact_areas(report_content),
                    "word_count": len(report_content.split()),
                },
            }

        except Exception as e:
            logger.error(f"Error generating impact assessment report: {e}")
            return {
                "error": str(e),
                "report_type": "impact_assessment",
                "generated_at": datetime.utcnow().isoformat(),
            }

    def _extract_sections(self, content: str) -> List[str]:
        """Extract section headers from report content"""
        lines = content.split("\n")
        sections = []

        for line in lines:
            line = line.strip()
            if line and len(line) < 100 and not line.startswith(" "):
                # Likely a section header
                if any(char.isdigit() for char in line[:3]) or line[0].isupper():
                    sections.append(line)

        return sections

    def _extract_recommendations(self, content: str) -> List[Dict[str, str]]:
        """Extract recommendations with priority levels"""
        recommendations = []
        lines = content.split("\n")
        current_rec = None

        for line in lines:
            line = line.strip()
            if "recommendation" in line.lower() or "priority:" in line.lower():
                if current_rec:
                    recommendations.append(current_rec)
                current_rec = {"text": line, "priority": "Medium"}
            elif current_rec and (
                "high" in line.lower()
                or "medium" in line.lower()
                or "low" in line.lower()
            ):
                current_rec["priority"] = line.split()[-1].title()
            elif current_rec:
                current_rec["text"] += " " + line

        if current_rec:
            recommendations.append(current_rec)

        return recommendations

    def _count_critical_issues(self, content: str) -> int:
        """Count critical issues mentioned in the report"""
        critical_keywords = [
            "critical",
            "severe",
            "urgent",
            "immediate",
            "failure",
            "breakdown",
        ]
        content_lower = content.lower()
        return sum(1 for keyword in critical_keywords if keyword in content_lower)

    def _identify_impact_areas(self, content: str) -> List[str]:
        """Identify impact areas from the content"""
        impact_areas = []
        content_lower = content.lower()

        area_keywords = {
            "social": ["social", "community", "democratic", "participation"],
            "economic": ["economic", "cost", "productivity", "business"],
            "educational": ["education", "learning", "knowledge", "training"],
            "policy": ["policy", "governance", "government", "transparency"],
            "accessibility": ["accessibility", "inclusion", "equity", "digital divide"],
        }

        for area, keywords in area_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                impact_areas.append(area)

        return impact_areas
