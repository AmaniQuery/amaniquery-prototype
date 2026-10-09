#!/usr/bin/env python3
"""
Example usage of AmaniQuery Research and Report Generation with Gemini AI

This script demonstrates how to use the new research and report generation
capabilities for user research purposes.
"""

import os
import json
import requests
from typing import Dict, List, Any

# Configuration
API_BASE_URL = "http://localhost:8000"


def check_research_status():
    """Check if research capabilities are available"""
    try:
        response = requests.get(f"{API_BASE_URL}/research/status")
        if response.status_code == 200:
            status = response.json()
            print("Research Status:")
            print(
                f"  Research Module: {'Available' if status['research_module_available'] else 'Not Available'}"
            )
            print(
                f"  Report Generator: {'Available' if status['report_generator_available'] else 'Not Available'}"
            )
            print(
                f"  Gemini API: {'Configured' if status['gemini_api_configured'] else 'Not Configured'}"
            )
            return status
        else:
            print(f"Failed to check status: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error checking research status: {e}")
        return None


def analyze_legal_query():
    """Analyze a legal query about Kenya's laws"""
    legal_query = """
    I am a small business owner in Nairobi. I recently received a notice from the Kenya Revenue Authority
    about tax compliance requirements for my retail shop. What are my rights and obligations under
    Kenyan tax law? Do I need to hire a tax consultant, or can I handle this myself?
    """

    context = {
        "user_type": "small_business_owner",
        "location": "Nairobi",
        "business_type": "retail_shop",
        "issue_type": "tax_compliance",
        "urgency": "medium",
    }

    try:
        response = requests.post(
            f"{API_BASE_URL}/research/analyze-legal-query",
            data={"query": legal_query, "context": json.dumps(context)},
        )

        if response.status_code == 200:
            result = response.json()
            print("Legal Query Analysis Completed!")
            print(f"Analysis generated at: {result['research_timestamp']}")
            print(f"Model used: {result['model_used']}")

            # Save analysis to file
            with open("legal_query_analysis.json", "w") as f:
                json.dump(result, f, indent=2)
            print("Analysis saved to legal_query_analysis.json")

            return result
        else:
            print(f"Analysis failed: {response.status_code} - {response.text}")
            return None

    except Exception as e:
        print(f"Error analyzing legal query: {e}")
        return None


def generate_legal_report(
    analysis_results: Dict[str, Any], focus: str = "comprehensive"
):
    """Generate a comprehensive legal report based on query analysis"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/research/generate-legal-report",
            data={
                "analysis_results": json.dumps(analysis_results),
                "report_focus": focus,
            },
        )

        if response.status_code == 200:
            result = response.json()
            print(f"Legal Report Generated ({focus} focus)!")

            # Save report to file
            filename = f"legal_report_{focus}.json"
            with open(filename, "w") as f:
                json.dump(result, f, indent=2)
            print(f"Report saved to {filename}")

            # Also save the report content as text
            if "report" in result and "content" in result["report"]:
                text_filename = f"legal_report_{focus}.txt"
                with open(text_filename, "w", encoding="utf-8") as f:
                    f.write(result["report"]["content"])
                print(f"Report content saved to {text_filename}")

            return result
        else:
            print(f"Report generation failed: {response.status_code} - {response.text}")
            return None

    except Exception as e:
        print(f"Error generating legal report: {e}")
        return None


def conduct_legal_research():
    """Conduct legal research on specific topics related to Kenya's laws"""
    legal_topics = [
        "Tax compliance requirements for small businesses in Kenya",
        "Consumer protection rights under Kenyan law",
        "Employment law obligations for employers in Kenya",
        "Environmental regulations for businesses in Kenya",
        "Intellectual property protection in Kenya",
    ]

    research_questions = [
        "What are the key legal requirements for small business tax compliance in Kenya?",
        "How does the Consumer Protection Act affect retail businesses in Kenya?",
        "What employment law obligations must employers comply with in Kenya?",
        "What environmental regulations apply to businesses operating in Kenya?",
        "How can businesses protect their intellectual property in Kenya?",
    ]

    try:
        response = requests.post(
            f"{API_BASE_URL}/research/legal-research",
            data={
                "legal_topics": json.dumps(legal_topics),
                "research_questions": json.dumps(research_questions),
            },
        )

        if response.status_code == 200:
            result = response.json()
            print("Legal Research Completed!")

            # Save results
            with open("legal_research_results.json", "w") as f:
                json.dump(result, f, indent=2)
            print("Research saved to legal_research_results.json")

            return result
        else:
            print(f"Legal research failed: {response.status_code} - {response.text}")
            return None

    except Exception as e:
        print(f"Error conducting legal research: {e}")
        return None


def generate_legal_query_report():
    """Generate a legal query report"""
    # First, let's create a sample analysis result
    sample_analysis = {
        "query_interpretation": {
            "main_question": "Tax compliance rights and obligations for small business owner",
            "key_concerns": [
                "Tax obligations",
                "Rights as taxpayer",
                "Need for professional help",
            ],
            "legal_areas": ["Tax law", "Administrative law", "Business regulation"],
        },
        "applicable_laws": {
            "primary_laws": [
                "Income Tax Act (Cap 470)",
                "Value Added Tax Act (Cap 476)",
                "Tax Procedures Act",
            ],
            "regulatory_bodies": ["Kenya Revenue Authority (KRA)"],
            "relevant_case_law": ["Recent High Court decisions on taxpayer rights"],
        },
        "legal_analysis": {
            "tax_obligations": [
                "File tax returns annually",
                "Pay taxes on time",
                "Maintain proper records",
            ],
            "taxpayer_rights": [
                "Right to fair administration",
                "Right to appeal decisions",
                "Right to information",
            ],
            "compliance_requirements": [
                "Register for taxes",
                "Obtain PIN number",
                "Keep financial records",
            ],
        },
        "practical_guidance": {
            "immediate_steps": [
                "Register with KRA if not done",
                "Obtain KRA PIN",
                "Organize financial records",
            ],
            "recommended_actions": [
                "Consider consulting a tax professional",
                "Set up proper accounting system",
            ],
            "resources": [
                "KRA website",
                "Tax help desks",
                "Business registration offices",
            ],
        },
    }

    try:
        response = requests.post(
            f"{API_BASE_URL}/reports/legal-query",
            data={"query_analysis": json.dumps(sample_analysis)},
        )

        if response.status_code == 200:
            result = response.json()
            print("Legal Query Report Generated!")

            # Save report
            with open("legal_query_report.json", "w") as f:
                json.dump(result, f, indent=2)
            print("Report saved to legal_query_report.json")

            return result
        else:
            print(f"Report generation failed: {response.status_code} - {response.text}")
            return None

    except Exception as e:
        print(f"Error generating legal query report: {e}")
        return None


def generate_pdf_report(
    analysis_results: Dict[str, Any], report_title: str = "Legal Research Report"
):
    """Generate a PDF report from legal analysis results"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/research/generate-pdf-report",
            data={
                "analysis_results": json.dumps(analysis_results),
                "report_title": report_title,
            },
        )

        if response.status_code == 200:
            # Save PDF file
            filename = f"{report_title.replace(' ', '_')}.pdf"
            with open(filename, "wb") as f:
                f.write(response.content)
            print(f"PDF Report Generated and saved as {filename}")
            return True
        else:
            print(f"PDF generation failed: {response.status_code} - {response.text}")
            return False

    except Exception as e:
        print(f"Error generating PDF report: {e}")
        return False


def generate_word_report(
    analysis_results: Dict[str, Any], report_title: str = "Legal Research Report"
):
    """Generate a Word document report from legal analysis results"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/research/generate-word-report",
            data={
                "analysis_results": json.dumps(analysis_results),
                "report_title": report_title,
            },
        )

        if response.status_code == 200:
            # Save Word document
            filename = f"{report_title.replace(' ', '_')}.docx"
            with open(filename, "wb") as f:
                f.write(response.content)
            print(f"Word Document Generated and saved as {filename}")
            return True
        else:
            print(
                f"Word document generation failed: {response.status_code} - {response.text}"
            )
            return False

    except Exception as e:
        print(f"Error generating Word document: {e}")
        return False
    """Generate a project overview report"""
    project_data = {
        "name": "AmaniQuery - Kenya Laws Intelligence",
        "tagline": "Democratizing Access to Kenyan Legal Information Through AI",
        "version": "1.0.0",
        "mission": "To make Kenyan laws, parliamentary proceedings, and constitutional rights accessible to every citizen through advanced AI technology",
        "core_focus": "Kenya Laws Intelligence",
        "target_country": "Kenya",
        "key_objectives": [
            "Provide instant access to Kenyan statutes, case law, and constitutional provisions",
            "Enable constitutional compliance analysis for legislation and policies",
            "Facilitate public understanding of legal and parliamentary processes",
            "Support legal professionals with AI-powered research tools",
            "Promote transparency and accountability in Kenya's legal system",
        ],
        "modules": [
            {
                "name": "NiruSpider",
                "purpose": "Comprehensive crawling of Kenyan legal sources",
                "key_sources": [
                    "Kenya Law Reports",
                    "Parliament of Kenya",
                    "Kenya Gazette",
                    "Court judgments",
                ],
            },
            {
                "name": "NiruParser",
                "purpose": "Legal document processing and AI analysis",
                "capabilities": [
                    "PDF text extraction",
                    "Legal text chunking",
                    "Constitutional alignment analysis",
                ],
            },
            {
                "name": "NiruDB",
                "purpose": "Intelligent legal knowledge base",
                "features": [
                    "Vector embeddings",
                    "Semantic search",
                    "Legal precedent linking",
                ],
            },
            {
                "name": "NiruAPI",
                "purpose": "AI-powered legal intelligence API",
                "models": [
                    "Moonshot AI",
                    "Gemini AI",
                    "Constitutional analysis engine",
                ],
            },
            {
                "name": "NiruShare",
                "purpose": "Legal knowledge dissemination",
                "channels": ["Social media", "SMS alerts", "Public dashboards"],
            },
        ],
        "key_features": [
            "Constitutional alignment checking for bills and policies",
            "Real-time parliamentary monitoring and bill tracking",
            "AI-powered legal research and document analysis",
            "SMS-based legal queries for mobile access",
            "Multi-language support (English, Swahili)",
            "Source-cited answers with legal references",
            "Public legal education tools",
            "Legal compliance monitoring dashboard",
        ],
        "impact_areas": [
            "Legal Access & Transparency",
            "Constitutional Rights Education",
            "Legal Professional Productivity",
            "Policy Compliance Monitoring",
            "Public Legal Literacy",
            "Judicial Efficiency",
            "Anti-Corruption Tools",
        ],
        "metrics": {
            "legal_documents_indexed": 50000,
            "constitutional_articles_analyzed": 264,
            "daily_legal_queries": 1200,
            "active_legal_professionals": 1500,
            "citizen_users": 25000,
            "average_response_time": "0.8s",
            "constitutional_alignment_accuracy": "94%",
        },
        "technologies": [
            "Python",
            "FastAPI",
            "Next.js",
            "ChromaDB",
            "Moonshot AI",
            "Gemini AI",
            "Scrapy",
            "PostgreSQL",
            "Docker",
            "Kubernetes",
            "SMS Gateway",
        ],
        "competitive_advantages": [
            "Specialized focus on Kenyan legal system",
            "Constitutional AI analysis capabilities",
            "Multi-channel access (web, mobile, SMS)",
            "Real-time legal monitoring",
            "Local language support",
            "Integration with official legal sources",
        ],
    }

    try:
        response = requests.post(
            f"{API_BASE_URL}/reports/project-overview",
            data={"project_data": json.dumps(project_data)},
        )

        if response.status_code == 200:
            result = response.json()
            print("Project Overview Report Generated!")

            # Save report
            with open("project_overview_report.json", "w") as f:
                json.dump(result, f, indent=2)
            print("Report saved to project_overview_report.json")

            return result
        else:
            print(f"Report generation failed: {response.status_code} - {response.text}")
            return None

    except Exception as e:
        print(f"Error generating project overview report: {e}")
        return None


def main():
    """Main demonstration function"""
    print("AmaniQuery Legal Research & Report Generation Demo")
    print("=" * 50)

    # Check if research capabilities are available
    status = check_research_status()
    if not status or not status.get("research_module_available"):
        print(
            "❌ Research module not available. Please ensure GEMINI_API_KEY is configured."
        )
        return

    print("\n✅ Research capabilities are available!")
    print()

    # Step 1: Analyze a legal query
    print("Step 1: Analyzing Legal Query...")
    analysis = analyze_legal_query()
    if not analysis:
        return

    print()

    # Step 2: Generate comprehensive legal report
    print("Step 2: Generating Comprehensive Legal Report...")
    report = generate_legal_report(analysis, "comprehensive")
    if not report:
        return

    print()

    # Step 3: Conduct legal research on business topics
    print("Step 3: Conducting Legal Research on Business Topics...")
    legal_research = conduct_legal_research()
    if not legal_research:
        return

    print()

    # Step 4: Generate legal query report
    print("Step 4: Generating Legal Query Report...")
    query_report = generate_legal_query_report()
    if not query_report:
        return

    # Step 5: Generate PDF report
    print("Step 5: Generating PDF Report...")
    pdf_success = generate_pdf_report(analysis, "Tax Compliance Legal Analysis")
    if not pdf_success:
        print("⚠️ PDF generation failed, but continuing...")

    print()

    # Step 6: Generate Word document
    print("Step 6: Generating Word Document...")
    word_success = generate_word_report(analysis, "Tax Compliance Legal Analysis")
    if not word_success:
        print("⚠️ Word document generation failed, but continuing...")

    print()
    print("🎉 Legal research and report generation completed!")
    print(
        "Check the generated files for detailed legal analysis, reports, and documents:"
    )
    print("- legal_query_analysis.json (raw analysis data)")
    print("- legal_report_comprehensive.json (structured report)")
    print("- legal_research_results.json (research findings)")
    print("- legal_query_report.json (formatted report)")
    print("- Tax_Compliance_Legal_Analysis.pdf (PDF document)")
    print("- Tax_Compliance_Legal_Analysis.docx (Word document)")


if __name__ == "__main__":
    main()
