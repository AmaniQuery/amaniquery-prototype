"""
Email Service for sending emails via Gmail SMTP
Handles email verification, password reset, and general notifications
"""

import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional, Dict, List
import logging
import os
from datetime import datetime

logger = logging.getLogger(__name__)


class EmailService:
    """Service for sending emails via Gmail SMTP"""

    def __init__(
        self,
        smtp_server: str = "smtp.gmail.com",
        smtp_port: int = 587,
        sender_email: str = "amaniquery@gmail.com",
        sender_password: Optional[str] = None,
        sender_name: str = "AmaniQuery",
    ):
        """
        Initialize email service

        Args:
            smtp_server: SMTP server address (default: smtp.gmail.com)
            smtp_port: SMTP port (default: 587 for TLS)
            sender_email: Email address to send from
            sender_password: Gmail app password (from environment if not provided)
            sender_name: Display name for sender
        """
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.sender_email = sender_email
        self.sender_password = sender_password or os.getenv("GMAIL_APP_PASSWORD")
        self.sender_name = sender_name

        if not self.sender_password:
            logger.warning(
                "Gmail app password not configured. Email sending will fail."
            )

    def _create_message(
        self,
        to_email: str,
        subject: str,
        html_body: str,
        text_body: Optional[str] = None,
    ) -> MIMEMultipart:
        """Create email message"""
        message = MIMEMultipart("alternative")
        message["Subject"] = subject
        message["From"] = f"{self.sender_name} <{self.sender_email}>"
        message["To"] = to_email

        # Add text and HTML parts
        if text_body:
            text_part = MIMEText(text_body, "plain")
            message.attach(text_part)

        html_part = MIMEText(html_body, "html")
        message.attach(html_part)

        return message

    def _send_email(self, message: MIMEMultipart, to_email: str) -> bool:
        """Send email via SMTP"""
        if not self.sender_password:
            logger.error("Cannot send email: Gmail app password not configured")
            return False

        try:
            # Create secure connection
            context = ssl.create_default_context()

            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls(context=context)
                server.login(self.sender_email, self.sender_password)
                server.send_message(message)

            logger.info(f"Email sent successfully to {to_email}")
            return True

        except smtplib.SMTPAuthenticationError as e:
            logger.error(f"SMTP authentication failed: {e}")
            return False
        except smtplib.SMTPException as e:
            logger.error(f"SMTP error occurred: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to send email to {to_email}: {e}")
            return False

    def send_verification_email(
        self,
        to_email: str,
        verification_token: str,
        verification_url: Optional[str] = None,
    ) -> bool:
        """
        Send email verification email

        Args:
            to_email: Recipient email address
            verification_token: Email verification token
            verification_url: Full verification URL (if None, will use token only)
        """
        if not verification_url:
            # Default verification URL format
            base_url = os.getenv("FRONTEND_URL", "http://localhost:3000")
            verification_url = (
                f"{base_url}/auth/verify-email?token={verification_token}"
            )

        subject = "Verify Your AmaniQuery Email Address"

        html_body = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Verify Your Email</title>
        </head>
        <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333; max-width: 600px; margin: 0 auto; padding: 20px;">
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px; text-align: center; border-radius: 10px 10px 0 0;">
                <h1 style="color: white; margin: 0;">AmaniQuery</h1>
            </div>
            <div style="background: #f9f9f9; padding: 30px; border-radius: 0 0 10px 10px;">
                <h2 style="color: #333; margin-top: 0;">Verify Your Email Address</h2>
                <p>Thank you for signing up for AmaniQuery! Please verify your email address by clicking the button below:</p>
                <div style="text-align: center; margin: 30px 0;">
                    <a href="{verification_url}" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 12px 30px; text-decoration: none; border-radius: 5px; display: inline-block; font-weight: bold;">Verify Email</a>
                </div>
                <p style="color: #666; font-size: 14px;">Or copy and paste this link into your browser:</p>
                <p style="color: #667eea; word-break: break-all; font-size: 12px;">{verification_url}</p>
                <p style="color: #666; font-size: 14px; margin-top: 30px;">This link will expire in 24 hours.</p>
                <p style="color: #666; font-size: 14px;">If you didn't create an account with AmaniQuery, please ignore this email.</p>
            </div>
            <div style="text-align: center; margin-top: 20px; color: #999; font-size: 12px;">
                <p>&copy; {datetime.now().year} AmaniQuery. All rights reserved.</p>
            </div>
        </body>
        </html>
        """

        text_body = f"""
        Verify Your Email Address
        
        Thank you for signing up for AmaniQuery! Please verify your email address by visiting the following link:
        
        {verification_url}
        
        This link will expire in 24 hours.
        
        If you didn't create an account with AmaniQuery, please ignore this email.
        
        © {datetime.now().year} AmaniQuery. All rights reserved.
        """

        message = self._create_message(to_email, subject, html_body, text_body)
        return self._send_email(message, to_email)

    def send_password_reset_email(
        self, to_email: str, reset_token: str, reset_url: Optional[str] = None
    ) -> bool:
        """
        Send password reset email

        Args:
            to_email: Recipient email address
            reset_token: Password reset token
            reset_url: Full reset URL (if None, will use token only)
        """
        if not reset_url:
            base_url = os.getenv("FRONTEND_URL", "http://localhost:3000")
            reset_url = f"{base_url}/auth/reset-password?token={reset_token}"

        subject = "Reset Your AmaniQuery Password"

        html_body = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Reset Your Password</title>
        </head>
        <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333; max-width: 600px; margin: 0 auto; padding: 20px;">
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px; text-align: center; border-radius: 10px 10px 0 0;">
                <h1 style="color: white; margin: 0;">AmaniQuery</h1>
            </div>
            <div style="background: #f9f9f9; padding: 30px; border-radius: 0 0 10px 10px;">
                <h2 style="color: #333; margin-top: 0;">Reset Your Password</h2>
                <p>We received a request to reset your password. Click the button below to create a new password:</p>
                <div style="text-align: center; margin: 30px 0;">
                    <a href="{reset_url}" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 12px 30px; text-decoration: none; border-radius: 5px; display: inline-block; font-weight: bold;">Reset Password</a>
                </div>
                <p style="color: #666; font-size: 14px;">Or copy and paste this link into your browser:</p>
                <p style="color: #667eea; word-break: break-all; font-size: 12px;">{reset_url}</p>
                <p style="color: #666; font-size: 14px; margin-top: 30px;">This link will expire in 1 hour.</p>
                <p style="color: #ff6b6b; font-size: 14px; font-weight: bold;">If you didn't request a password reset, please ignore this email and your password will remain unchanged.</p>
            </div>
            <div style="text-align: center; margin-top: 20px; color: #999; font-size: 12px;">
                <p>&copy; {datetime.now().year} AmaniQuery. All rights reserved.</p>
            </div>
        </body>
        </html>
        """

        text_body = f"""
        Reset Your Password
        
        We received a request to reset your password. Visit the following link to create a new password:
        
        {reset_url}
        
        This link will expire in 1 hour.
        
        If you didn't request a password reset, please ignore this email and your password will remain unchanged.
        
        © {datetime.now().year} AmaniQuery. All rights reserved.
        """

        message = self._create_message(to_email, subject, html_body, text_body)
        return self._send_email(message, to_email)

    def send_notification_email(
        self,
        to_email: str,
        subject: str,
        message: str,
        html_message: Optional[str] = None,
    ) -> bool:
        """
        Send general notification email

        Args:
            to_email: Recipient email address
            subject: Email subject
            message: Plain text message
            html_message: HTML message (optional)
        """
        html_body = html_message or f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
        </head>
        <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333; max-width: 600px; margin: 0 auto; padding: 20px;">
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px; text-align: center; border-radius: 10px 10px 0 0;">
                <h1 style="color: white; margin: 0;">AmaniQuery</h1>
            </div>
            <div style="background: #f9f9f9; padding: 30px; border-radius: 0 0 10px 10px;">
                <div style="white-space: pre-wrap;">{message}</div>
            </div>
            <div style="text-align: center; margin-top: 20px; color: #999; font-size: 12px;">
                <p>&copy; {datetime.now().year} AmaniQuery. All rights reserved.</p>
            </div>
        </body>
        </html>
        """

        email_message = self._create_message(to_email, subject, html_body, message)
        return self._send_email(email_message, to_email)

    def send_welcome_email(self, to_email: str, name: str) -> bool:
        """Send welcome email to new users"""
        subject = "Welcome to AmaniQuery!"

        html_body = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Welcome to AmaniQuery</title>
        </head>
        <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333; max-width: 600px; margin: 0 auto; padding: 20px;">
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px; text-align: center; border-radius: 10px 10px 0 0;">
                <h1 style="color: white; margin: 0;">Welcome to AmaniQuery!</h1>
            </div>
            <div style="background: #f9f9f9; padding: 30px; border-radius: 0 0 10px 10px;">
                <h2 style="color: #333; margin-top: 0;">Hello {name}!</h2>
                <p>Thank you for joining AmaniQuery - your AI-powered platform for Kenyan legal, parliamentary, and news intelligence.</p>
                <p>You can now:</p>
                <ul>
                    <li>Ask questions about Kenyan law and legislation</li>
                    <li>Search parliamentary proceedings</li>
                    <li>Get insights from news articles</li>
                    <li>Analyze constitutional alignment</li>
                </ul>
                <div style="text-align: center; margin: 30px 0;">
                    <a href="{os.getenv('FRONTEND_URL', 'http://localhost:3000')}/chat" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 12px 30px; text-decoration: none; border-radius: 5px; display: inline-block; font-weight: bold;">Get Started</a>
                </div>
                <p style="color: #666; font-size: 14px;">If you have any questions, feel free to reach out to our support team.</p>
            </div>
            <div style="text-align: center; margin-top: 20px; color: #999; font-size: 12px;">
                <p>&copy; {datetime.now().year} AmaniQuery. All rights reserved.</p>
            </div>
        </body>
        </html>
        """

        text_body = f"""
        Welcome to AmaniQuery!
        
        Hello {name}!
        
        Thank you for joining AmaniQuery - your AI-powered platform for Kenyan legal, parliamentary, and news intelligence.
        
        You can now:
        - Ask questions about Kenyan law and legislation
        - Search parliamentary proceedings
        - Get insights from news articles
        - Analyze constitutional alignment
        
        Get started: {os.getenv('FRONTEND_URL', 'http://localhost:3000')}/chat
        
        If you have any questions, feel free to reach out to our support team.
        
        © {datetime.now().year} AmaniQuery. All rights reserved.
        """

        message = self._create_message(to_email, subject, html_body, text_body)
        return self._send_email(message, to_email)


def get_email_service() -> EmailService:
    """Get email service instance with configuration from environment"""
    return EmailService(
        smtp_server=os.getenv("SMTP_SERVER", "smtp.gmail.com"),
        smtp_port=int(os.getenv("SMTP_PORT", "587")),
        sender_email=os.getenv("GMAIL_SENDER_EMAIL", "amaniquery@gmail.com"),
        sender_password=os.getenv("GMAIL_APP_PASSWORD"),
        sender_name=os.getenv("EMAIL_SENDER_NAME", "AmaniQuery"),
    )
