"""Look up the numeric Instagram user id for a @handle, using the bot account.

The number it prints is what goes in ALLOWED_SENDER_ID — pass your *personal*
handle (the account you'll share posts FROM). Logging in as the bot account here
also completes any first-time verification challenge and saves the session.

    python -m bot.whoami your_personal_handle
"""
from __future__ import annotations

import os
import sys

from . import config  # importing loads .env into the environment
from .sources.instagram import login_client


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print("usage: python -m bot.whoami <instagram_handle>", file=sys.stderr)
        return 2
    handle = argv[0].lstrip("@")

    username = os.environ.get("IG_USERNAME")
    password = os.environ.get("IG_PASSWORD")
    if not (username and password):
        print(
            "IG_USERNAME and IG_PASSWORD must be set in .env first.", file=sys.stderr
        )
        return 2

    client = login_client(username, password, config.DATA_DIR / "session.json")
    try:
        client.dump_settings(config.DATA_DIR / "session.json")
    except Exception:  # noqa: BLE001
        pass

    user_id = client.user_id_from_username(handle)
    print(user_id)
    print(
        f"\nAdd this to your .env:\nALLOWED_SENDER_ID={user_id}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
