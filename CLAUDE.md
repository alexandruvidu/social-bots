# CLAUDE.md

See `README.md` for setup, architecture, and the data flow. This file covers
operational details that aren't obvious from the code.

## Nothing here may log in to Instagram

Logging in to Instagram's private API (`instagrapi`) is what got the bot
account forcibly logged out everywhere — on the Pi *and* on the phone. That is
the step before suspension. The whole pipeline was restructured so that no
automatic code path can log in. **Don't reintroduce one, and don't run one by
hand to debug.**

- `bot.webhook` — Flask server (`python -m bot.webhook`), the live path. Uses
  the official Instagram Messaging API only. Stays running, handles events as
  they arrive via a Cloudflare Tunnel (`scripts/run_tunnel.sh`). It never logs
  in: everything it analyzes comes from the webhook's own attachment payload.
- `bot.run` — the old cron batch poller. It still drives `instagrapi`, so it is
  now **guarded**: `python -m bot.run` exits immediately unless
  `ALLOW_PRIVATE_API=1` is set in the environment. Treat that override as a
  break-glass switch, not a debugging convenience.
- `bot/sources/instagram.py` — kept on disk so the change is reversible, but
  **unwired**. Nothing in the live path imports it (`bot.run` imports it lazily,
  inside the guarded function). Don't wire it back in.

Check `ps aux | grep bot.webhook` / `crontab -l` to see what is actually
running before assuming anything — this has changed over time as the project
evolved from cron to webhook.

## How a share actually gets resolved

The official Messaging API delivers an attachment payload, and that is the only
content the bot gets — there is no caption, geotag, or comment fetch any more.
`bot/run.py:_process_post` walks this ladder:

1. `analyze_media(post.media_url, post.media_kind)` — the media the webhook
   handed us, sent to Gemini. `bot/sources/instagram_api.py` classifies
   `attachment.payload.url` by host: an `instagram.com` permalink is a link with
   no media, anything else (`lookaside.fbsbx.com`) is a signed CDN URL we can
   download. Attachment type maps to `media_kind`: `ig_reel`/`video` → video,
   `image` → image, `share` → unknown, resolved from the download's own
   `Content-Type` (never a HEAD sniff — signed CDNs reject those).
2. `extract(caption, location, comments)` — only runs when there is text, which
   in practice means the dormant `instagrapi` path. Live posts skip it.
3. Ask the user (`ASK_TEXT`) and write a `pending` row.

A **screenshot is the guaranteed floor** of the system: `image` attachments are
always real CDN URLs, so when the user answers an ask with a screenshot it goes
to `handle_media_replies` → `analyze_image` and is saved against the pending
row's link with `source_field="screenshot"`.

Routing note (`bot/webhook.py:_match_pending_ask`): a screenshot is normally
sent as a plain message with no quote, so an attachment is treated as an answer
when it either quotes an open ask *or* is type `image` while an ask is open in
that thread. Shared posts arrive as `share`/`ig_reel`/`video`, so they never
take that branch. Accepted trade-off: an unrelated photo sent while an ask is
open gets consumed as the answer.

Several asks can be outstanding per thread — `pending` is keyed on
`(platform, ask_msg_id)`, and `get_pending(platform, thread_id)` returns the
most recent open row for use when a reply doesn't quote anything.

## Webhook process management (current quirk)

Systemd units exist (`social-bots-webhook.service`, `social-bots-tunnel.service`)
but are **disabled**; `systemctl status social-bots-webhook` will show
`inactive (dead)` even while the bot is live. In practice the webhook has been
started manually:

```bash
nohup .venv/bin/python -m bot.webhook > /tmp/webhook.log 2>&1 & disown
```

`app.run()` has no reloader, so **code changes require a manual restart** —
editing `bot/extract.py`, `bot/webhook.py`, etc. has no effect on the
running process until you kill it and relaunch the command above. Check
`ps aux | grep bot.webhook` for the PID, `kill <pid>`, then relaunch. Logs land
in `/tmp/webhook.log` (not `data/bot.log`, which is only written by `bot.run`).

If asked to make this robust long-term, the systemd units are the intended
path (`sudo systemctl enable --now social-bots-webhook social-bots-tunnel`) —
but don't switch management modes without checking with the user first, since
the manual process may be intentional for now.

## Dead configuration — don't tune it expecting an effect

`COMMENTS_LIMIT`, `COMMENTS_FETCH_LIMIT`, and `IG_SESSION_PATH` are still read
by `bot/config.py`, and `_select_comments` / `_looks_like_place_name` still
exist in `bot/sources/instagram.py` with tests in
`tests/test_instagram_source.py`. **None of it runs in the live path** —
comments were lost as a signal when enrichment was removed. If a destination
goes unfound, the lever is the media analysis (`analyze_media`, `VIDEO_SYSTEM`,
`SCREENSHOT_SYSTEM` in `bot/extract.py`), not comment selection.

`IG_USERNAME` / `IG_PASSWORD` are likewise optional now (`str | None`); the bot
boots with no Instagram credentials on disk at all.

## Working agreements

- No paid APIs — this project runs entirely on Gemini's free tier
  (`GEMINI_API_KEY`); don't introduce a paid model or service as a fix.
- Don't commit markdown/doc files (including plans under `docs/`) — leave
  those local.
- Don't run `git commit` unless explicitly asked — leave committing to the
  user.
- Never commit `.env`, `data/session.json`, or `data/db.sqlite` — already in
  `.gitignore`; double check before any `git add`.

## Tests

```bash
.venv/bin/pytest tests/ -v                 # offline, no network
RUN_LIVE_TESTS=1 .venv/bin/pytest tests/test_extract.py::test_live_extraction  # hits Gemini
```

Every test is offline: all Gemini and HTTP calls are mocked, nothing logs in to
Instagram, and no test may open `data/db.sqlite` (point `DB_PATH` at a tmp path
in any test that constructs a real `Store`).

**There is no safe way to inspect a real post's caption/comments any more, and
you should not go looking for one** — every such route goes through
`instagrapi` login. To debug extraction, work from what the bot actually
receives: the raw attachment payload is logged at INFO by
`build_post_from_event`, so `/tmp/webhook.log` has the real `payload.url` from
the last share. Feed that URL to `analyze_media` directly, or reproduce with a
downloaded file and `analyze_image` / `analyze_video`.

Sending replies via the official Graph API (`InstagramAPISource.reply`) is a
first-class Meta integration and is safe to use for diagnostics.
