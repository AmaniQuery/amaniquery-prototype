"""
Research Router - Research and report generation endpoints for AmaniQuery
"""

import json
import asyncio
import os
from typing import Optional, Dict, List, Any
from pathlib import Path
from fastapi import APIRouter, HTTPException, Form
from fastapi.responses import Response, FileResponse
from loguru import logger
from pydantic import BaseModel

router = APIRouter(prefix="/research", tags=["Research"])

# Reports router for report generation
reports_router = APIRouter(prefix="/reports", tags=["Reports"])


# =============================================================================
# DEPENDENCIES - State container to avoid global variable issues
# =============================================================================


class ResearchRouterState:
    """State container for research router dependencies"""

    agentic_research_module = None
    research_module = None
    report_generator = None
    cache_manager = None
    chat_manager = None
    research_bundle_service = None


_state = ResearchRouterState()


def get_research_module():
    """Get the active research module (agentic if available, else legacy)"""
    if _state.agentic_research_module is not None:
        return _state.agentic_research_module
    if _state.research_module is not None:
        return _state.research_module
    raise HTTPException(
        status_code=503,
        detail="Research module not available. Ensure API keys are configured.",
    )


def get_report_generator():
    """Get the report generator instance"""
    if _state.report_generator is None:
        raise HTTPException(
            status_code=503,
            detail="Report generator not available. Ensure GEMINI_API_KEY is configured.",
        )
    return _state.report_generator


def save_query_to_chat(session_id: str, query: str, result: Dict):
    """Helper function to save query and response to chat database"""
    if _state.chat_manager is None or not session_id:
        return

    try:
        session = _state.chat_manager.get_session(session_id)
        if not session:
            return

        chat_result = {
            "answer": result.get("analysis", result.get("summary", str(result))),
            "sources": result.get("sources", []),
            "retrieved_chunks": result.get("chunks_used", 0),
            "model_used": result.get("model_used", "gemini"),
        }

        _state.chat_manager.add_message(
            session_id=session_id, content=query, role="user"
        )

        _state.chat_manager.add_message(
            session_id=session_id,
            content=chat_result["answer"],
            role="assistant",
            token_count=chat_result["retrieved_chunks"],
            model_used=chat_result["model_used"],
            sources=chat_result["sources"],
        )
    except Exception as e:
        logger.warning(f"Failed to save query to chat: {e}")


# =============================================================================
# RESEARCH ENDPOINTS
# =============================================================================


@router.post("/analyze-legal-query")
async def analyze_legal_query(
    query: str = Form(...),
    context: Optional[str] = Form(None),
    session_id: Optional[str] = Form(None),
):
    """
    Analyze a legal query about Kenya's laws using Gemini AI

    **Parameters:**
    - query: The legal question or query to analyze
    - context: Optional additional context about the query (JSON string)
    - session_id: Optional chat session ID to save messages

    **Returns:**
    - Comprehensive legal analysis
    """
    module = get_research_module()

    try:
        context_data = None
        if context:
            try:
                context_data = json.loads(context)
            except json.JSONDecodeError:
                context_data = {"additional_info": context}

        # Use async method for agentic module
        if hasattr(module, "analyze_legal_query") and asyncio.iscoroutinefunction(
            module.analyze_legal_query
        ):
            result = await module.analyze_legal_query(query, context_data)
        else:
            result = module.analyze_legal_query(query, context_data)

        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])

        if session_id:
            save_query_to_chat(session_id, query, result)

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in legal query analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate-legal-report")
async def generate_legal_report(
    analysis_results: str = Form(...), report_focus: str = Form("comprehensive")
):
    """
    Generate a comprehensive legal report based on query analysis

    **Parameters:**
    - analysis_results: JSON string of analysis results from /research/analyze-legal-query
    - report_focus: Type of legal focus (comprehensive, constitutional, criminal, civil, administrative)

    **Returns:**
    - Structured legal report
    """
    module = get_research_module()

    try:
        analysis_data = json.loads(analysis_results)
        result = module.generate_legal_report(analysis_data, report_focus)

        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])

        return result

    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in analysis_results")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating legal report: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/legal-research")
async def conduct_legal_research(
    legal_topics: str = Form(...),
    research_questions: str = Form(...),
    session_id: Optional[str] = Form(None),
):
    """
    Conduct legal research on specific topics related to Kenya's laws

    **Parameters:**
    - legal_topics: JSON string array of legal topics to research
    - research_questions: JSON string array of specific research questions
    - session_id: Optional chat session ID to save messages

    **Returns:**
    - Legal research findings
    """
    module = get_research_module()

    try:
        topics = json.loads(legal_topics)
        questions = json.loads(research_questions)

        if not isinstance(topics, list) or not isinstance(questions, list):
            raise HTTPException(
                status_code=400,
                detail="legal_topics and research_questions must be JSON arrays",
            )

        result = module.conduct_legal_research(topics, questions)

        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])

        if session_id:
            query_text = f"Research on topics: {', '.join(topics)}. Questions: {', '.join(questions)}"
            save_query_to_chat(session_id, query_text, result)

        return result

    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON format: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error conducting legal research: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate-pdf-report")
async def generate_pdf_report(
    analysis_results: str = Form(...), report_title: str = Form("Legal Research Report")
):
    """
    Generate a PDF report from legal analysis results

    **Returns:**
    - PDF file as downloadable content
    """
    bundle_service = _state.research_bundle_service
    if bundle_service is None:
        raise HTTPException(
            status_code=503, detail="Research bundle service not available"
        )

    try:
        analysis_data = json.loads(analysis_results)
        query = analysis_data.get("original_query", "Legal Research Report")
        bundle_dict = await bundle_service.conduct_research(
            query=query,
            context=analysis_data,
            generate_pdf=True,
            generate_docx=False,
        )
        if bundle_dict.get("status") != "completed":
            raise RuntimeError(bundle_dict.get("error", "Research failed"))
        bundle = bundle_service.get_bundle(bundle_dict["bundle_id"])
        if not bundle or not bundle.pdf_path:
            raise RuntimeError("PDF generation failed")

        return FileResponse(
            path=bundle.pdf_path,
            media_type="application/pdf",
            filename=f"{report_title.replace(' ', '_')}.pdf",
        )
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in analysis_results")
    except Exception as e:
        logger.error(f"Error generating PDF report: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate-word-report")
async def generate_word_report(
    analysis_results: str = Form(...), report_title: str = Form("Legal Research Report")
):
    """
    Generate a Word document report from legal analysis results

    **Returns:**
    - Word document (.docx) as downloadable content
    """
    bundle_service = _state.research_bundle_service
    if bundle_service is None:
        raise HTTPException(
            status_code=503, detail="Research bundle service not available"
        )

    try:
        analysis_data = json.loads(analysis_results)
        query = analysis_data.get("original_query", "Legal Research Report")
        bundle_dict = await bundle_service.conduct_research(
            query=query,
            context=analysis_data,
            generate_pdf=False,
            generate_docx=True,
        )
        if bundle_dict.get("status") != "completed":
            raise RuntimeError(bundle_dict.get("error", "Research failed"))
        bundle = bundle_service.get_bundle(bundle_dict["bundle_id"])
        if not bundle or not bundle.docx_path:
            raise RuntimeError("DOCX generation failed")

        return FileResponse(
            path=bundle.docx_path,
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            filename=f"{report_title.replace(' ', '_')}.docx",
        )
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in analysis_results")
    except Exception as e:
        logger.error(f"Error generating Word report: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def get_research_status():
    """Get the status of research and report generation capabilities"""
    module_available = (
        _state.research_module is not None or _state.agentic_research_module is not None
    )

    return {
        "research_module_available": module_available,
        "agentic_research_available": _state.agentic_research_module is not None,
        "report_generator_available": _state.report_generator is not None,
        "bundle_service_available": _state.research_bundle_service is not None,
        "gemini_api_configured": bool(os.getenv("GEMINI_API_KEY")),
        "available_endpoints": (
            [
                "/research/analyze-legal-query",
                "/research/generate-legal-report",
                "/research/legal-research",
                "/research/conduct",
                "/research/download/{bundle_id}/{format}",
                "/research/generate-pdf-report",
                "/research/generate-word-report",
                "/reports/legal-query",
                "/reports/legal-research",
                "/reports/constitutional-law",
                "/reports/compliance",
                "/reports/technical-audit",
                "/reports/impact-assessment",
            ]
            if module_available
            else []
        ),
    }


# =============================================================================
# UNIFIED RESEARCH FLOW ENDPOINTS
# =============================================================================


@router.post("/conduct")
async def conduct_research(
    query: str = Form(...),
    context: Optional[str] = Form(None),
    session_id: Optional[str] = Form(None),
    generate_pdf: bool = Form(True),
    generate_docx: bool = Form(True),
):
    """
    Conduct full research and get results bundled with downloadable documents.

    **Parameters:**
    - query: The research question or query
    - context: Optional additional context (JSON string)
    - session_id: Optional chat session ID for saving to chat
    - generate_pdf: Whether to auto-generate PDF (default: true)
    - generate_docx: Whether to auto-generate DOCX (default: true)

    **Returns:**
    - Research bundle with analysis, download URLs for PDF/DOCX
    """
    bundle_service = _state.research_bundle_service
    if bundle_service is None:
        raise HTTPException(
            status_code=503, detail="Research bundle service not available"
        )

    try:
        context_data = None
        if context:
            try:
                context_data = json.loads(context)
            except json.JSONDecodeError:
                context_data = {"additional_info": context}

        result = await bundle_service.conduct_research(
            query=query,
            context=context_data,
            session_id=session_id,
            generate_pdf=generate_pdf,
            generate_docx=generate_docx,
        )

        if session_id:
            module = _state.agentic_research_module or _state.research_module
            if module:
                analysis = result.get("analysis", {})
                chat_result = {
                    "answer": analysis.get("query_interpretation", str(result)),
                    "sources": result.get("sources", []),
                    "model_used": "research-bundle",
                }
                save_query_to_chat(session_id, query, chat_result)

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in unified research: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/download/{bundle_id}/{format}")
async def download_research_bundle(bundle_id: str, format: str):
    """
    Download a generated research document.

    **Parameters:**
    - bundle_id: The bundle ID from /research/conduct response
    - format: "pdf" or "docx"

    **Returns:**
    - The requested document file
    """
    bundle_service = _state.research_bundle_service
    if bundle_service is None:
        raise HTTPException(
            status_code=503, detail="Research bundle service not available"
        )

    file_path = bundle_service.get_bundle_download_path(bundle_id, format)
    if file_path is None:
        raise HTTPException(status_code=404, detail="Bundle not found or expired")

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File no longer available")

    media_types = {
        "pdf": "application/pdf",
        "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    }
    media_type = media_types.get(format, "application/octet-stream")
    filename = f"research_{bundle_id}.{format}"

    return FileResponse(
        path=file_path,
        media_type=media_type,
        filename=filename,
    )


# =============================================================================
# REPORT GENERATION ENDPOINTS
# =============================================================================


@reports_router.post("/legal-query")
async def generate_legal_query_report(query_analysis: str = Form(...)):
    """
    Generate a comprehensive legal query report
    """
    report_generator = get_report_generator()

    try:
        analysis = json.loads(query_analysis)
        result = report_generator.generate_legal_query_report(analysis)

        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])

        return result

    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in query_analysis")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating legal query report: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@reports_router.post("/legal-research")
async def generate_legal_research_report(
    research_data: str = Form(...), research_findings: str = Form(...)
):
    """
    Generate a legal research report
    """
    report_generator = get_report_generator()

    try:
        research_info = json.loads(research_data)
        findings = json.loads(research_findings)

        result = report_generator.generate_legal_research_report(
            research_info, findings
        )

        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])

        return result

    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON format: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating legal research report: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@reports_router.post("/constitutional-law")
async def generate_constitutional_law_report(constitutional_analysis: str = Form(...)):
    """
    Generate a constitutional law report
    """
    report_generator = get_report_generator()

    try:
        analysis = json.loads(constitutional_analysis)
        result = report_generator.generate_constitutional_law_report(analysis)

        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])

        return result

    except json.JSONDecodeError:
        raise HTTPException(
            status_code=400, detail="Invalid JSON in constitutional_analysis"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating constitutional law report: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@reports_router.post("/compliance")
async def generate_compliance_report(
    legal_requirements: str = Form(...), compliance_data: str = Form(...)
):
    """
    Generate a legal compliance report
    """
    report_generator = get_report_generator()

    try:
        requirements = json.loads(legal_requirements)
        compliance = json.loads(compliance_data)

        result = report_generator.generate_compliance_report(requirements, compliance)

        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])

        return result

    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON format: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating compliance report: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@reports_router.post("/technical-audit")
async def generate_technical_audit_report(
    system_metrics: str = Form(...), performance_data: str = Form(...)
):
    """
    Generate a technical audit report
    """
    report_generator = get_report_generator()

    try:
        metrics = json.loads(system_metrics)
        performance = json.loads(performance_data)

        result = report_generator.generate_technical_audit_report(metrics, performance)

        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])

        return result

    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON format: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating technical audit report: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@reports_router.post("/impact-assessment")
async def generate_impact_assessment_report(
    usage_data: str = Form(...), impact_metrics: str = Form(...)
):
    """
    Generate an impact assessment report
    """
    report_generator = get_report_generator()

    try:
        usage = json.loads(usage_data)
        impact = json.loads(impact_metrics)

        result = report_generator.generate_impact_assessment_report(usage, impact)

        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])

        return result

    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON format: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating impact assessment report: {e}")
        raise HTTPException(status_code=500, detail=str(e))
