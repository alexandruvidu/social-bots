# Travel-Destination DM Bot

Share a travel reel/post into a dedicated Instagram account's DMs, and this bot
extracts the destination (e.g. "Kyoto, Japan") and saves it — with the post link
— to a local SQLite database. A personal travel wishlist, built from DMs.

## How it works

1. A long-running **webhook server** (`bot.webhook`, official Instagram
   Messaging API) receives DM events as they happen and processes new shared
   posts **from your account only**. (There's also a `bot.run` batch/cron mode
   that polls DMs instead — see "Alternative: cron batch mode" below — but the
   webhook is what's actually used day to day.)
2. For each public Instagram permalink it reads the **caption, geotag, and top
   comments** with the no-login `instagram_downloader.py` / Instaloader layer;
   posts that deny anonymous access fall back to the webhook media attachment.
3. It sends those to **Google Gemini** (`gemini-2.5-flash`, free tier) which
   returns a normalized destination or "none".
4. Found → saved to SQLite (deduped on the post link).
   Not found → the bot DMs you asking for the place; your **next text reply** in
   that thread gets saved.

Instagram-only for now; the `Source` abstraction (`bot/sources/base.py`) is built
so TikTok can be added later (no DM-read API exists for TikTok today).

> The live webhook never logs in through `instagrapi`. Instaloader still reads
> public Instagram pages unofficially, so Instagram may rate-limit or deny an
> individual lookup; the bot then continues with media analysis or asks you.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill it in
```

`.env` needs your **personal** account's numeric IG user id (`ALLOWED_SENDER_ID`
— only DMs from this id are trusted), the official Messaging API credentials, and a free
`GEMINI_API_KEY` from https://aistudio.google.com/app/apikey (no card required).

### Configuring Instagram

1. Configure the bot account as a Business/Creator account and connect it to
   the Meta app as described below.
2. Set `ALLOWED_SENDER_ID` to your personal account's IGSID (the webhook logs
   untrusted sender IDs, which makes it easy to confirm).
3. `INSTALOADER_ENRICHMENT=1` is enabled by default. Set it to `0` if you want
   a strictly official-API-only runtime.

## Run (webhook mode — the live setup)

Requires a Business/Creator IG account for the bot, a Meta Developer App
(`IG_ACCESS_TOKEN`, `IG_APP_SECRET`, `IG_USER_ID`, `META_APP_ID`,
`WEBHOOK_VERIFY_TOKEN` in `.env`), and a public URL pointing at the server —
`scripts/run_tunnel.sh` opens a Cloudflare quick tunnel and registers it with
Meta automatically.

```bash
.venv/bin/python -m bot.webhook        # Flask server, stays running
scripts/run_tunnel.sh                  # in another shell: tunnel + webhook registration
```

Systemd units for both (`social-bots-webhook.service`,
`social-bots-tunnel.service`) exist under `/etc/systemd/system/` for running
this unattended; enable with `systemctl enable --now social-bots-webhook
social-bots-tunnel` if you want it to survive reboots.

No reloader — restart the process (kill + relaunch) after pulling code
changes.

## Alternative: cron batch mode

`bot.run` polls DMs instead of receiving webhook events — useful if you'd
rather not expose a public endpoint, or as a fallback. Not needed if the
webhook is running.

```bash
python -m bot.run
```

```cron
*/15 * * * * cd /home/veedoo/social_bots && /home/veedoo/social_bots/.venv/bin/python -m bot.run >> data/bot.log 2>&1
```

## Inspect saved destinations

```bash
sqlite3 data/db.sqlite "SELECT destination, link, source_field, created_at FROM destinations ORDER BY created_at DESC;"
```

## Tests

```bash
pip install pytest
pytest                       # offline: storage + extraction logic (no network)
RUN_LIVE_TESTS=1 pytest tests/test_extract.py::test_live_extraction   # hits Gemini
```

## Layout

```
bot/
  config.py                env/config loading
  store.py                 SQLite: destinations, processed, pending
  extract.py               Gemini structured-output extraction
  sources/base.py          Source protocol + SharedPost/TextReply
  sources/instagram.py     dormant instagrapi batch implementation (explicit opt-in only)
  ../instagram_downloader.py  no-login public permalink metadata reader
  sources/instagram_api.py official Messaging API implementation (used by webhook mode)
  run.py                   shared handle_posts/handle_replies + batch orchestrator (python -m bot.run)
  webhook.py               Flask server for webhook mode (python -m bot.webhook)
scripts/run_tunnel.sh      Cloudflare tunnel + webhook URL registration, for webhook mode
tests/                     offline unit tests + an opt-in live test
```
