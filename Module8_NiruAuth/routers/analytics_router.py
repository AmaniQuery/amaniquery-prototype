"""
Analytics Router
Provides usage statistics and analytics
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from sqlalchemy import func, and_
from datetime import datetime, timedelta
from typing import List, Optional

from ..models.pydantic_models import UsageStats, UsageLogResponse, AnalyticsDashboard
from ..dependencies import get_db, get_current_user, require_admin
from ..models.auth_models import User, Integration, UsageLog
from ..authorization.user_role_manager import UserRoleManager

router = APIRouter(prefix="/api/v1/auth/analytics", tags=["Analytics"])


@router.get("/integrations/{integration_id}/usage", response_model=UsageStats)
async def get_integration_usage(
    integration_id: str,
    days: int = Query(30, ge=1, le=365),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get usage statistics for integration"""
    # Verify ownership
    integration = db.query(Integration).filter(Integration.id == integration_id).first()
    if not integration:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Integration not found"
        )

    if integration.owner_user_id != user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized"
        )

    # Calculate date range
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)

    # Query usage logs
    logs = (
        db.query(UsageLog)
        .filter(
            and_(
                UsageLog.integration_id == integration_id,
                UsageLog.timestamp >= start_date,
                UsageLog.timestamp <= end_date,
            )
        )
        .all()
    )

    # Calculate stats
    total_requests = len(logs)
    total_tokens = sum(log.tokens_used for log in logs)
    total_cost = sum(log.cost for log in logs)
    avg_response_time = (
        sum(log.response_time_ms or 0 for log in logs) / total_requests
        if total_requests > 0
        else 0
    )

    # Get today's requests
    today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    requests_today = (
        db.query(UsageLog)
        .filter(
            and_(
                UsageLog.integration_id == integration_id,
                UsageLog.timestamp >= today_start,
            )
        )
        .count()
    )

    # Get this week's requests
    week_start = today_start - timedelta(days=today_start.weekday())
    requests_this_week = (
        db.query(UsageLog)
        .filter(
            and_(
                UsageLog.integration_id == integration_id,
                UsageLog.timestamp >= week_start,
            )
        )
        .count()
    )

    # Get this month's requests
    month_start = today_start.replace(day=1)
    requests_this_month = (
        db.query(UsageLog)
        .filter(
            and_(
                UsageLog.integration_id == integration_id,
                UsageLog.timestamp >= month_start,
            )
        )
        .count()
    )

    # Top endpoints
    endpoint_stats = (
        db.query(
            UsageLog.endpoint,
            func.count(UsageLog.id).label("count"),
            func.avg(UsageLog.response_time_ms).label("avg_time"),
        )
        .filter(
            and_(
                UsageLog.integration_id == integration_id,
                UsageLog.timestamp >= start_date,
            )
        )
        .group_by(UsageLog.endpoint)
        .order_by(func.count(UsageLog.id).desc())
        .limit(10)
        .all()
    )

    top_endpoints = [
        {
            "endpoint": stat[0],
            "count": stat[1],
            "avg_response_time_ms": float(stat[2] or 0),
        }
        for stat in endpoint_stats
    ]

    return UsageStats(
        total_requests=total_requests,
        total_tokens=total_tokens,
        total_cost=total_cost,
        avg_response_time_ms=avg_response_time,
        requests_today=requests_today,
        requests_this_week=requests_this_week,
        requests_this_month=requests_this_month,
        top_endpoints=top_endpoints,
    )


@router.get("/users/usage", response_model=UsageStats)
async def get_user_usage(
    days: int = Query(30, ge=1, le=365),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get usage statistics for current user"""
    # Calculate date range
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)

    # Query usage logs
    logs = (
        db.query(UsageLog)
        .filter(
            and_(
                UsageLog.user_id == user.id,
                UsageLog.timestamp >= start_date,
                UsageLog.timestamp <= end_date,
            )
        )
        .all()
    )

    # Calculate stats (similar to integration usage)
    total_requests = len(logs)
    total_tokens = sum(log.tokens_used for log in logs)
    total_cost = sum(log.cost for log in logs)
    avg_response_time = (
        sum(log.response_time_ms or 0 for log in logs) / total_requests
        if total_requests > 0
        else 0
    )

    today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    requests_today = (
        db.query(UsageLog)
        .filter(and_(UsageLog.user_id == user.id, UsageLog.timestamp >= today_start))
        .count()
    )

    week_start = today_start - timedelta(days=today_start.weekday())
    requests_this_week = (
        db.query(UsageLog)
        .filter(and_(UsageLog.user_id == user.id, UsageLog.timestamp >= week_start))
        .count()
    )

    month_start = today_start.replace(day=1)
    requests_this_month = (
        db.query(UsageLog)
        .filter(and_(UsageLog.user_id == user.id, UsageLog.timestamp >= month_start))
        .count()
    )

    endpoint_stats = (
        db.query(
            UsageLog.endpoint,
            func.count(UsageLog.id).label("count"),
            func.avg(UsageLog.response_time_ms).label("avg_time"),
        )
        .filter(and_(UsageLog.user_id == user.id, UsageLog.timestamp >= start_date))
        .group_by(UsageLog.endpoint)
        .order_by(func.count(UsageLog.id).desc())
        .limit(10)
        .all()
    )

    top_endpoints = [
        {
            "endpoint": stat[0],
            "count": stat[1],
            "avg_response_time_ms": float(stat[2] or 0),
        }
        for stat in endpoint_stats
    ]

    return UsageStats(
        total_requests=total_requests,
        total_tokens=total_tokens,
        total_cost=total_cost,
        avg_response_time_ms=avg_response_time,
        requests_today=requests_today,
        requests_this_week=requests_this_week,
        requests_this_month=requests_this_month,
        top_endpoints=top_endpoints,
    )


@router.get("/dashboard", response_model=AnalyticsDashboard)
async def get_analytics_dashboard(
    admin: User = Depends(require_admin), db: Session = Depends(get_db)
):
    """Get analytics dashboard (admin only)"""
    # Total counts
    total_users = db.query(User).count()
    total_integrations = db.query(Integration).count()

    # Get API keys count
    from ..models.auth_models import APIKey

    total_api_keys = db.query(APIKey).filter(APIKey.is_active == True).count()

    # Today's stats
    today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    today_logs = db.query(UsageLog).filter(UsageLog.timestamp >= today_start).all()
    total_requests_today = len(today_logs)
    total_cost_today = sum(log.cost for log in today_logs)

    # This week's stats
    week_start = today_start - timedelta(days=today_start.weekday())
    week_logs = db.query(UsageLog).filter(UsageLog.timestamp >= week_start).all()
    total_requests_this_week = len(week_logs)
    total_cost_this_week = sum(log.cost for log in week_logs)

    # This month's stats
    month_start = today_start.replace(day=1)
    month_logs = db.query(UsageLog).filter(UsageLog.timestamp >= month_start).all()
    total_requests_this_month = len(month_logs)
    total_cost_this_month = sum(log.cost for log in month_logs)

    # Top users
    user_stats = (
        db.query(
            UsageLog.user_id,
            func.count(UsageLog.id).label("count"),
            func.sum(UsageLog.cost).label("total_cost"),
        )
        .filter(UsageLog.user_id.isnot(None), UsageLog.timestamp >= month_start)
        .group_by(UsageLog.user_id)
        .order_by(func.count(UsageLog.id).desc())
        .limit(10)
        .all()
    )

    top_users = [
        {"user_id": stat[0], "requests": stat[1], "total_cost": float(stat[2] or 0)}
        for stat in user_stats
    ]

    # Top integrations
    integration_stats = (
        db.query(
            UsageLog.integration_id,
            func.count(UsageLog.id).label("count"),
            func.sum(UsageLog.cost).label("total_cost"),
        )
        .filter(UsageLog.integration_id.isnot(None), UsageLog.timestamp >= month_start)
        .group_by(UsageLog.integration_id)
        .order_by(func.count(UsageLog.id).desc())
        .limit(10)
        .all()
    )

    top_integrations = [
        {
            "integration_id": stat[0],
            "requests": stat[1],
            "total_cost": float(stat[2] or 0),
        }
        for stat in integration_stats
    ]

    # Top endpoints
    endpoint_stats = (
        db.query(
            UsageLog.endpoint,
            func.count(UsageLog.id).label("count"),
            func.avg(UsageLog.response_time_ms).label("avg_time"),
        )
        .filter(UsageLog.timestamp >= month_start)
        .group_by(UsageLog.endpoint)
        .order_by(func.count(UsageLog.id).desc())
        .limit(10)
        .all()
    )

    top_endpoints = [
        {
            "endpoint": stat[0],
            "count": stat[1],
            "avg_response_time_ms": float(stat[2] or 0),
        }
        for stat in endpoint_stats
    ]

    # Requests over time (last 30 days)
    requests_over_time = []
    for i in range(30):
        day_start = today_start - timedelta(days=i)
        day_end = day_start + timedelta(days=1)
        day_count = (
            db.query(UsageLog)
            .filter(and_(UsageLog.timestamp >= day_start, UsageLog.timestamp < day_end))
            .count()
        )
        requests_over_time.append({"date": day_start.isoformat(), "count": day_count})

    requests_over_time.reverse()

    return AnalyticsDashboard(
        total_users=total_users,
        total_integrations=total_integrations,
        total_api_keys=total_api_keys,
        total_requests_today=total_requests_today,
        total_requests_this_week=total_requests_this_week,
        total_requests_this_month=total_requests_this_month,
        total_cost_today=total_cost_today,
        total_cost_this_week=total_cost_this_week,
        total_cost_this_month=total_cost_this_month,
        top_users=top_users,
        top_integrations=top_integrations,
        top_endpoints=top_endpoints,
        requests_over_time=requests_over_time,
    )
