"""
Notification Service - Manage subscriptions and send notifications
"""

import os
import sys
from pathlib import Path
from typing import List, Optional, Dict
from datetime import datetime, time, timedelta
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from Module3_NiruDB.notification_models import (
    NotificationSubscription,
    SubscriptionCreate,
    SubscriptionUpdate,
    SubscriptionResponse,
    create_database_engine,
    create_tables,
    get_db_session,
)
from Module4_NiruAPI.services.talksasa_service import TalksasaNotificationService


class NotificationService:
    """Service for managing notification subscriptions and sending notifications"""

    def __init__(self, database_url: Optional[str] = None, config_manager=None):
        """
        Initialize notification service

        Args:
            database_url: PostgreSQL connection URL
            config_manager: ConfigManager instance for API tokens
        """
        if database_url is None:
            database_url = os.getenv(
                "DATABASE_URL", "postgresql://localhost/amaniquery"
            )

        # Handle Neon connection pooling
        if "neon.tech" in database_url and "pooler" in database_url:
            unpooled_url = os.getenv("DATABASE_URL_UNPOOLED")
            if unpooled_url:
                database_url = unpooled_url

        self.database_url = database_url
        self.engine = create_database_engine(database_url)
        create_tables(self.engine)

        # Initialize Talksasa service
        api_token = None
        sender_id = None
        if config_manager:
            api_token = config_manager.get_config("TALKSASA_API_TOKEN")
            sender_id = config_manager.get_config("TALKSASA_SENDER_ID")

        self.talksasa_service = TalksasaNotificationService(
            api_token=api_token, sender_id=sender_id
        )
        logger.info("Notification service initialized")

    def _get_db_session(self):
        """Get database session"""
        return get_db_session(self.engine)

    def subscribe(self, subscription: SubscriptionCreate) -> SubscriptionResponse:
        """
        Create or update a notification subscription

        Args:
            subscription: Subscription data

        Returns:
            SubscriptionResponse
        """
        db = self._get_db_session()
        try:
            # Check if subscription exists
            existing = (
                db.query(NotificationSubscription)
                .filter_by(phone_number=subscription.phone_number)
                .first()
            )

            # Parse digest_time if provided
            digest_time_obj = None
            if subscription.digest_time:
                try:
                    hour, minute = subscription.digest_time.split(":")
                    digest_time_obj = time(int(hour), int(minute))
                except:
                    pass

            if existing:
                # Update existing subscription
                existing.notification_type = subscription.notification_type
                existing.schedule_type = subscription.schedule_type
                existing.digest_time = digest_time_obj
                existing.categories = subscription.categories
                existing.sources = subscription.sources
                existing.is_active = True
                existing.updated_at = datetime.utcnow()
                db.commit()
                db.refresh(existing)
                logger.info(f"Updated subscription for {subscription.phone_number}")
            else:
                # Create new subscription
                new_sub = NotificationSubscription(
                    phone_number=subscription.phone_number,
                    notification_type=subscription.notification_type,
                    schedule_type=subscription.schedule_type,
                    digest_time=digest_time_obj,
                    categories=subscription.categories,
                    sources=subscription.sources,
                    is_active=True,
                )
                db.add(new_sub)
                db.commit()
                db.refresh(new_sub)
                existing = new_sub
                logger.info(f"Created subscription for {subscription.phone_number}")

            # Convert to response
            return self._subscription_to_response(existing)

        except Exception as e:
            db.rollback()
            logger.error(f"Error subscribing: {e}")
            raise
        finally:
            db.close()

    def unsubscribe(self, phone_number: str) -> bool:
        """
        Deactivate a subscription

        Args:
            phone_number: Phone number to unsubscribe

        Returns:
            True if successful
        """
        db = self._get_db_session()
        try:
            subscription = (
                db.query(NotificationSubscription)
                .filter_by(phone_number=phone_number)
                .first()
            )

            if subscription:
                subscription.is_active = False
                subscription.updated_at = datetime.utcnow()
                db.commit()
                logger.info(f"Unsubscribed {phone_number}")
                return True
            return False

        except Exception as e:
            db.rollback()
            logger.error(f"Error unsubscribing: {e}")
            raise
        finally:
            db.close()

    def update_subscription(
        self, phone_number: str, update: SubscriptionUpdate
    ) -> Optional[SubscriptionResponse]:
        """
        Update subscription preferences

        Args:
            phone_number: Phone number
            update: Update data

        Returns:
            Updated SubscriptionResponse or None if not found
        """
        db = self._get_db_session()
        try:
            subscription = (
                db.query(NotificationSubscription)
                .filter_by(phone_number=phone_number)
                .first()
            )

            if not subscription:
                return None

            if update.notification_type is not None:
                subscription.notification_type = update.notification_type
            if update.schedule_type is not None:
                subscription.schedule_type = update.schedule_type
            if update.digest_time is not None:
                try:
                    hour, minute = update.digest_time.split(":")
                    subscription.digest_time = time(int(hour), int(minute))
                except:
                    pass
            if update.categories is not None:
                subscription.categories = update.categories
            if update.sources is not None:
                subscription.sources = update.sources
            if update.is_active is not None:
                subscription.is_active = update.is_active

            subscription.updated_at = datetime.utcnow()
            db.commit()
            db.refresh(subscription)

            logger.info(f"Updated subscription for {phone_number}")
            return self._subscription_to_response(subscription)

        except Exception as e:
            db.rollback()
            logger.error(f"Error updating subscription: {e}")
            raise
        finally:
            db.close()

    def get_subscription(self, phone_number: str) -> Optional[SubscriptionResponse]:
        """
        Get subscription by phone number

        Args:
            phone_number: Phone number

        Returns:
            SubscriptionResponse or None
        """
        db = self._get_db_session()
        try:
            subscription = (
                db.query(NotificationSubscription)
                .filter_by(phone_number=phone_number)
                .first()
            )

            if subscription:
                return self._subscription_to_response(subscription)
            return None

        except Exception as e:
            logger.error(f"Error getting subscription: {e}")
            return None
        finally:
            db.close()

    def get_active_subscriptions(
        self,
        schedule_type: Optional[str] = None,
        categories: Optional[List[str]] = None,
        sources: Optional[List[str]] = None,
    ) -> List[SubscriptionResponse]:
        """
        Get active subscriptions with optional filters

        Args:
            schedule_type: Filter by schedule type
            categories: Filter by categories (subscriptions that match any)
            sources: Filter by sources (subscriptions that match any)

        Returns:
            List of SubscriptionResponse
        """
        db = self._get_db_session()
        try:
            query = db.query(NotificationSubscription).filter_by(is_active=True)

            if schedule_type:
                query = query.filter_by(schedule_type=schedule_type)

            subscriptions = query.all()
            results = []

            for sub in subscriptions:
                # Filter by categories if provided
                if categories:
                    sub_categories = sub.categories or []
                    if not any(cat in sub_categories for cat in categories):
                        # If subscription has category filters, check if they match
                        if sub_categories and not any(
                            cat in categories for cat in sub_categories
                        ):
                            continue

                # Filter by sources if provided
                if sources:
                    sub_sources = sub.sources or []
                    if not any(src in sub_sources for src in sources):
                        # If subscription has source filters, check if they match
                        if sub_sources and not any(
                            src in sources for src in sub_sources
                        ):
                            continue

                results.append(self._subscription_to_response(sub))

            return results

        except Exception as e:
            logger.error(f"Error getting subscriptions: {e}")
            return []
        finally:
            db.close()

    def send_article_notification(self, article: Dict) -> int:
        """
        Send notification for a new article to matching subscribers

        Args:
            article: Article dict with title, source_name, category, url, etc.

        Returns:
            Number of notifications sent
        """
        # Get subscribers who want immediate notifications
        subscribers = self.get_active_subscriptions(
            schedule_type="immediate",
            categories=[article.get("category")] if article.get("category") else None,
            sources=(
                [article.get("source_name")] if article.get("source_name") else None
            ),
        )

        sent_count = 0
        message = self._format_article_message(article)

        for subscriber in subscribers:
            # Check if article matches subscriber's filters
            if not self._article_matches_subscription(article, subscriber):
                continue

            try:
                result = self.talksasa_service.send_notification(
                    recipient=subscriber.phone_number,
                    message=message,
                    notification_type=subscriber.notification_type,
                )

                if result.get("status") == "success":
                    sent_count += 1
                else:
                    logger.warning(
                        f"Failed to send notification to {subscriber.phone_number}: {result.get('message')}"
                    )

            except Exception as e:
                logger.error(
                    f"Error sending notification to {subscriber.phone_number}: {e}"
                )

        logger.info(
            f"Sent {sent_count} article notifications for: {article.get('title', 'Unknown')}"
        )
        return sent_count

    def send_digest_notifications(self) -> int:
        """
        Send daily digest to subscribers who opted for it

        Returns:
            Number of digests sent
        """
        from Module4_NiruAPI.services.news_service import NewsService

        # Get subscribers who want daily digest
        subscribers = self.get_active_subscriptions(schedule_type="daily_digest")

        if not subscribers:
            return 0

        news_service = NewsService()
        sent_count = 0
        current_time = datetime.utcnow().time()

        for subscriber in subscribers:
            # Check if it's time to send digest
            if subscriber.digest_time:
                try:
                    # Allow 30 minute window
                    digest_time = datetime.strptime(
                        subscriber.digest_time, "%H:%M"
                    ).time()
                    time_diff = abs(
                        (
                            datetime.combine(datetime.today(), current_time)
                            - datetime.combine(datetime.today(), digest_time)
                        ).total_seconds()
                        / 60
                    )
                    if time_diff > 30:
                        continue
                except:
                    # If time parsing fails, skip this subscriber
                    continue

            try:
                # Get articles from last 24 hours matching subscriber filters
                date_from = (datetime.utcnow() - timedelta(days=1)).isoformat()
                articles, _ = news_service.get_articles(
                    sources=subscriber.sources,
                    categories=subscriber.categories,
                    date_from=date_from,
                    limit=10,
                )

                if not articles:
                    continue

                message = self._format_digest_message(articles)

                result = self.talksasa_service.send_notification(
                    recipient=subscriber.phone_number,
                    message=message,
                    notification_type=subscriber.notification_type,
                )

                if result.get("status") == "success":
                    sent_count += 1
                else:
                    logger.warning(
                        f"Failed to send digest to {subscriber.phone_number}: {result.get('message')}"
                    )

            except Exception as e:
                logger.error(f"Error sending digest to {subscriber.phone_number}: {e}")

        logger.info(f"Sent {sent_count} daily digest notifications")
        return sent_count

    def _article_matches_subscription(
        self, article: Dict, subscription: SubscriptionResponse
    ) -> bool:
        """Check if article matches subscription filters"""
        # Check categories
        if subscription.categories:
            article_category = article.get("category")
            if article_category not in subscription.categories:
                return False

        # Check sources
        if subscription.sources:
            article_source = article.get("source_name")
            if article_source not in subscription.sources:
                return False

        return True

    def _format_article_message(self, article: Dict) -> str:
        """Format article as notification message (max 160 chars for SMS)"""
        title = article.get("title", "New Article")
        source = article.get("source_name", "News")
        category = article.get("category", "")
        url = article.get("url", "")

        # Truncate title if needed
        max_title_len = 80
        if len(title) > max_title_len:
            title = title[: max_title_len - 3] + "..."

        message = f"[NEWS] {title}\n"
        if category:
            message += f"Category: {category}\n"
        message += f"Source: {source}"

        # Try to add URL if space allows
        if url and len(message) + len(url) + 5 < 160:
            # Shorten URL if needed
            if len(url) > 30:
                url = url[:27] + "..."
            message += f"\n{url}"

        # Ensure message is within SMS limit
        if len(message) > 160:
            message = message[:157] + "..."

        return message

    def _format_digest_message(self, articles: List[Dict]) -> str:
        """Format daily digest message"""
        message = f"[NEWS] Daily News Digest ({len(articles)} articles)\n\n"

        for i, article in enumerate(articles[:5], 1):  # Limit to 5 articles
            title = article.get("title", "Article")
            if len(title) > 40:
                title = title[:37] + "..."
            message += f"{i}. {title}\n"

        if len(articles) > 5:
            message += f"\n...and {len(articles) - 5} more"

        # Ensure message is within reasonable length
        if len(message) > 500:
            message = message[:497] + "..."

        return message

    def _subscription_to_response(
        self, subscription: NotificationSubscription
    ) -> SubscriptionResponse:
        """Convert database model to response model"""
        digest_time_str = None
        if subscription.digest_time:
            digest_time_str = subscription.digest_time.strftime("%H:%M")

        return SubscriptionResponse(
            id=subscription.id,
            phone_number=subscription.phone_number,
            notification_type=subscription.notification_type,
            schedule_type=subscription.schedule_type,
            digest_time=digest_time_str,
            categories=subscription.categories,
            sources=subscription.sources,
            is_active=subscription.is_active,
            created_at=subscription.created_at,
            updated_at=subscription.updated_at,
        )
