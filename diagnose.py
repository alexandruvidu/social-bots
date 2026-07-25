"""Run as: python diagnose.py — dumps raw DM thread data to find why a post is missed."""
import os, sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))

from bot.config import Config
from bot.sources.instagram import login_client

cfg = Config.load()
client = login_client(cfg.ig_username, cfg.ig_password, cfg.session_path)

print(f"Logged in. Fetching up to 20 threads...\n")
threads = client.direct_threads(amount=20)
print(f"Found {len(threads)} threads.\n")

for thread in threads:
    thread_id = str(thread.id)
    msgs = getattr(thread, "messages", []) or []
    print(f"Thread {thread_id}: {len(msgs)} messages")
    for msg in msgs:
        user_id = str(getattr(msg, "user_id", ""))
        item_type = getattr(msg, "item_type", None)
        msg_id = str(msg.id)
        is_allowed = user_id == str(cfg.allowed_sender_id)
        marker = "*** ALLOWED SENDER ***" if is_allowed else f"(sender {user_id})"
        print(f"  msg {msg_id}  type={item_type!r}  {marker}")
        if is_allowed and item_type:
            from bot.sources.instagram import SHARE_ATTRS
            if item_type in SHARE_ATTRS:
                print(f"    -> RECOGNIZED share type, would be processed")
            elif item_type == "text":
                print(f"    -> text reply")
            else:
                print(f"    -> UNRECOGNIZED item_type — bot skips this!")
    print()

client.dump_settings(cfg.session_path)
