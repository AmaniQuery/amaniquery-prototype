"""
Spider Monitoring and Health Check System
Monitors spider performance, source reliability, and data quality
"""

import asyncio
import aiohttp
import time
from datetime import datetime
from typing import Dict, List, Tuple
from collections import defaultdict
import json
from pathlib import Path


class SpiderMonitor:
    """Monitor spider health and source reliability"""

    def __init__(self, output_dir: str = "logs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.results = defaultdict(dict)

    async def check_rss_feed(
        self, url: str, name: str, session: aiohttp.ClientSession
    ) -> Dict:
        """Check if RSS feed is accessible"""
        try:
            start_time = time.time()
            async with session.get(
                url, timeout=aiohttp.ClientTimeout(total=15)
            ) as response:
                elapsed = time.time() - start_time

                return {
                    "url": url,
                    "name": name,
                    "status": response.status,
                    "accessible": response.status == 200,
                    "response_time": round(elapsed, 2),
                    "content_type": response.headers.get("Content-Type", "unknown"),
                    "error": None,
                }
        except asyncio.TimeoutError:
            return {
                "url": url,
                "name": name,
                "status": 0,
                "accessible": False,
                "response_time": None,
                "content_type": None,
                "error": "Timeout",
            }
        except Exception as e:
            return {
                "url": url,
                "name": name,
                "status": 0,
                "accessible": False,
                "response_time": None,
                "content_type": None,
                "error": str(e),
            }

    async def check_all_sources(self) -> Dict:
        """Check all spider sources"""
        # News RSS feeds from updated news_rss_spider.py
        news_feeds = [
            ("https://nation.africa/kenya/news/rss", "Nation Africa - News"),
            ("https://nation.africa/kenya/politics/rss", "Nation Africa - Politics"),
            ("https://nation.africa/kenya/business/rss", "Nation Africa - Business"),
            (
                "https://www.standardmedia.co.ke/rss/headlines.php",
                "Standard Media - Headlines",
            ),
            ("https://www.standardmedia.co.ke/rss/kenya.php", "Standard Media - Kenya"),
            (
                "https://www.standardmedia.co.ke/rss/politics.php",
                "Standard Media - Politics",
            ),
            (
                "https://www.standardmedia.co.ke/rss/business.php",
                "Standard Media - Business",
            ),
            (
                "https://www.standardmedia.co.ke/rss/opinion.php",
                "Standard Media - Opinion",
            ),
            ("https://www.the-star.co.ke/rss/news", "The Star - News"),
            ("https://www.the-star.co.ke/rss/business", "The Star - Business"),
            ("https://www.the-star.co.ke/rss/opinion", "The Star - Opinion"),
            (
                "https://www.businessdailyafrica.com/bd/economy/rss",
                "Business Daily - Economy",
            ),
            (
                "https://www.businessdailyafrica.com/bd/corporate/companies/rss",
                "Business Daily - Corporate",
            ),
            (
                "https://www.theeastafrican.co.ke/tea/news/rss",
                "The East African - News",
            ),
            ("https://www.citizen.digital/news/feed/", "Citizen Digital"),
            ("https://www.ktnnews.com/feed/", "KTN News"),
            ("https://www.ntvkenya.co.ke/feed/", "NTV Kenya"),
            ("https://www.capitalfm.co.ke/news/feed/", "Capital FM News"),
            ("https://www.tuko.co.ke/feed/", "Tuko.co.ke"),
            ("https://www.pulselive.co.ke/news/feed/", "Pulse Live Kenya"),
            ("https://www.kenyans.co.ke/feed/", "Kenyans.co.ke"),
        ]

        # Government websites
        government_sites = [
            ("https://www.parliament.go.ke", "Parliament of Kenya"),
            ("http://kenyalaw.org", "Kenya Law"),
        ]

        results = {
            "timestamp": datetime.now().isoformat(),
            "news_feeds": [],
            "government_sites": [],
            "summary": {},
        }

        # Check RSS feeds
        async with aiohttp.ClientSession() as session:
            print("[CHECK] Checking RSS feeds...")
            tasks = [
                self.check_rss_feed(url, name, session) for url, name in news_feeds
            ]
            news_results = await asyncio.gather(*tasks)
            results["news_feeds"] = news_results

            print("[CHECK] Checking government sites...")
            tasks = [
                self.check_rss_feed(url, name, session)
                for url, name in government_sites
            ]
            gov_results = await asyncio.gather(*tasks)
            results["government_sites"] = gov_results

        # Calculate summary
        total_feeds = len(news_results)
        accessible_feeds = sum(1 for r in news_results if r["accessible"])
        avg_response_time = sum(
            r["response_time"] for r in news_results if r["response_time"]
        ) / max(accessible_feeds, 1)

        total_gov = len(gov_results)
        accessible_gov = sum(1 for r in gov_results if r["accessible"])

        results["summary"] = {
            "total_rss_feeds": total_feeds,
            "accessible_rss_feeds": accessible_feeds,
            "rss_success_rate": round(accessible_feeds / total_feeds * 100, 1),
            "avg_rss_response_time": round(avg_response_time, 2),
            "total_government_sites": total_gov,
            "accessible_government_sites": accessible_gov,
            "gov_success_rate": (
                round(accessible_gov / total_gov * 100, 1) if total_gov > 0 else 0
            ),
        }

        self.results = results
        return results

    def generate_report(self, save_to_file: bool = True) -> str:
        """Generate a human-readable health report"""
        if not self.results:
            return "No results available. Run check_all_sources() first."

        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("SPIDER SOURCE HEALTH REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {self.results['timestamp']}")
        report_lines.append("")

        # Summary
        summary = self.results["summary"]
        report_lines.append("[SUMMARY]")
        report_lines.append("-" * 80)
        report_lines.append(
            f"RSS Feeds: {summary['accessible_rss_feeds']}/{summary['total_rss_feeds']} accessible ({summary['rss_success_rate']}%)"
        )
        report_lines.append(f"Avg Response Time: {summary['avg_rss_response_time']}s")
        report_lines.append(
            f"Government Sites: {summary['accessible_government_sites']}/{summary['total_government_sites']} accessible ({summary['gov_success_rate']}%)"
        )
        report_lines.append("")

        # RSS Feed Details
        report_lines.append("[NEWS] RSS FEEDS")
        report_lines.append("-" * 80)

        accessible = [f for f in self.results["news_feeds"] if f["accessible"]]
        failed = [f for f in self.results["news_feeds"] if not f["accessible"]]

        if accessible:
            report_lines.append(f"\n[OK] Accessible ({len(accessible)}):")
            for feed in accessible:
                report_lines.append(f"  - {feed['name']}")
                report_lines.append(f"    {feed['url']}")
                report_lines.append(
                    f"    Status: {feed['status']} | Response Time: {feed['response_time']}s"
                )

        if failed:
            report_lines.append(f"\n[FAIL] Failed ({len(failed)}):")
            for feed in failed:
                report_lines.append(f"  - {feed['name']}")
                report_lines.append(f"    {feed['url']}")
                report_lines.append(
                    f"    Error: {feed['error'] or 'HTTP ' + str(feed['status'])}"
                )

        # Government Sites
        report_lines.append("\n")
        report_lines.append("[GOV] GOVERNMENT SITES")
        report_lines.append("-" * 80)
        for site in self.results["government_sites"]:
            status_icon = "[OK]" if site["accessible"] else "[FAIL]"
            report_lines.append(f"{status_icon} {site['name']}")
            report_lines.append(f"   {site['url']}")
            if site["accessible"]:
                report_lines.append(
                    f"   Status: {site['status']} | Response Time: {site['response_time']}s"
                )
            else:
                report_lines.append(
                    f"   Error: {site['error'] or 'HTTP ' + str(site['status'])}"
                )

        report_lines.append("")
        report_lines.append("=" * 80)

        report = "\n".join(report_lines)

        if save_to_file:
            filename = (
                self.output_dir
                / f"spider_health_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            )
            with open(filename, "w", encoding="utf-8") as f:
                f.write(report)

            # Also save JSON
            json_filename = (
                self.output_dir
                / f"spider_health_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            with open(json_filename, "w", encoding="utf-8") as f:
                json.dump(self.results, f, indent=2)

            print(f"\n[NOTE] Report saved to: {filename}")
            print(f"[INFO] JSON data saved to: {json_filename}")

        return report


async def main():
    """Main entry point for spider monitoring"""
    print("[MONITOR] AmaniQuery Spider Health Monitor")
    print("=" * 80)
    print()

    monitor = SpiderMonitor()

    print("Starting health checks...")
    await monitor.check_all_sources()

    print("\n" + "=" * 80)
    report = monitor.generate_report(save_to_file=True)
    print(report)

    # Print recommendations
    summary = monitor.results["summary"]
    print("\n[TIP] RECOMMENDATIONS")
    print("-" * 80)

    if summary["rss_success_rate"] < 90:
        print(
            "[WARN] RSS feed success rate is below 90%. Review and update failed feeds."
        )
    else:
        print("[OK] RSS feed success rate is good!")

    if summary["avg_rss_response_time"] > 3.0:
        print("[WARN] Average response time is high. Consider adjusting timeouts.")
    else:
        print("[OK] Response times are acceptable.")

    failed_feeds = [f for f in monitor.results["news_feeds"] if not f["accessible"]]
    if failed_feeds:
        print(f"\n[ACTION] Action needed for {len(failed_feeds)} failed feeds:")
        for feed in failed_feeds:
            print(f"   - {feed['name']}: {feed['url']}")

    print()


if __name__ == "__main__":
    asyncio.run(main())
