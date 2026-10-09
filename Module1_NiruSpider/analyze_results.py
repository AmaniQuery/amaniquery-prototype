import json
import glob

files = sorted(glob.glob("logs/spider_health_*.json"), reverse=True)
d = json.load(open(files[0]))
s = d["summary"]

print("=" * 80)
print("LATEST MONITORING RESULTS")
print("=" * 80)
print(f'RSS Success Rate: {s["rss_success_rate"]}%')
print(f'Accessible: {s["accessible_rss_feeds"]}/{s["total_rss_feeds"]}')
print(f'Avg Response: {s["avg_rss_response_time"]}s')

acc = [f for f in d["news_feeds"] if f["accessible"]]
fail = [f for f in d["news_feeds"] if not f["accessible"]]

print(f"\n[OK] Working Feeds ({len(acc)}):")
for f in acc:
    print(f'  - {f["name"]} ({f["response_time"]}s)')

print(f"\n[FAIL] Failed Feeds ({len(fail)}):")
for f in fail:
    err = f.get("error") if f.get("error") else f["status"]
    print(f'  - {f["name"]}: {err}')

print("\n" + "=" * 80)
print(f'IMPROVEMENT: {len(acc)}/{len(acc)+len(fail)} = {s["rss_success_rate"]}%')
