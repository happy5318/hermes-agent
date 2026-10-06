"""Discord and Slack approval buttons must resolve THEIR queued request (#124974).

``TurnRunner._approval_notify_sync`` now forwards the queued entry's
``approval_request_id`` inside the card metadata, and every bundled adapter that
reads it resolves that specific entry instead of the FIFO-oldest one. Discord and
Slack shipped without wiring on the follow-up pass: both called
``resolve_gateway_approval(session_key, choice)`` — the two-argument form that
falls back to ``queue.pop(0)`` — so with two pending approvals in one session,
tapping the NEWEST card still answered the OLDEST request and the newest stayed
blocked for the full ``approvals.timeout``.

These tests drive the real send/click handlers and assert the request id crosses
the wire in both directions:

- **Discord** — the ``ExecApprovalView`` carries the id it was built with, and
  ``_resolve`` passes it to the resolver.
- **Slack** — the button value embeds the id behind a unit-separator marker, and
  ``_handle_approval_action`` splits it back out before resolving.

Both are fail-closed: an id that matches nothing in the queue resolves zero
entries rather than degrading to FIFO, so an unanswerable card cannot approve an
unrelated request. A legacy card with no id stays on the FIFO path unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Ensure the repo root is importable
# ---------------------------------------------------------------------------
_repo = str(Path(__file__).resolve().parents[2])
if _repo not in sys.path:
    sys.path.insert(0, _repo)

RID = "df87ed20af3c4b1e9d5a6c7f8b901234"

# The marker that carries the id inside the Slack button value. Session keys are
# ``platform:agent:...`` shaped and never contain U+001F, so the split is safe.
SLACK_RID_SEP = "\x1frid="


# ===========================================================================
# Discord
# ===========================================================================


def _import_discord_adapter():
    """Import the adapter with a usable ``discord`` module.

    ``test_discord_imports.py`` re-imports the adapter with discord simulated
    missing (``DISCORD_AVAILABLE`` False), which leaves a cached module without
    the view classes. Re-import a healthy copy when that happened, so these
    tests do not depend on module import order."""
    import importlib

    import plugins.platforms.discord.adapter as dmod

    if getattr(dmod, "DISCORD_AVAILABLE", True) and hasattr(dmod, "ExecApprovalView"):
        return dmod

    for _name in [k for k in list(sys.modules)
                  if k == "plugins.platforms.discord.adapter" or k.startswith("plugins.platforms.discord.")]:
        del sys.modules[_name]
    dmod = importlib.import_module("plugins.platforms.discord.adapter")
    return dmod


def _ensure_view_remove_item():
    """A view that drops buttons needs ``remove_item``.

    Patched on the *instance* a test builds rather than on the shared
    ``discord.ui.View`` class, so no other test file can observe the change.
    Idempotent per instance."""
    def _remove_item(self, item):
        for attr in ("children", "_children"):
            container = getattr(self, attr, None)
            if isinstance(container, list):
                try:
                    container.remove(item)
                    return
                except ValueError:
                    pass

    return _remove_item


def _make_discord_view(**kwargs):
    """Build the real ExecApprovalView, giving it an instance-level remove_item."""
    dmod = _import_discord_adapter()

    view = dmod.ExecApprovalView(
        session_key="agent:main:discord:dm:1", allowed_user_ids={"1"}, **kwargs)
    if not hasattr(view, "remove_item"):
        view.remove_item = _ensure_view_remove_item().__get__(view, type(view))
    return view


class TestDiscordApprovalRequestId:
    """The view that renders a card must hold (and resolve) that card's id."""

    def test_view_carries_the_forwarded_request_id(self):
        view = _make_discord_view(request_id=RID)
        assert view.request_id == RID

    def test_resolve_passes_the_request_id_to_the_resolver(self):
        """Tapping the button must target ITS entry, not the FIFO-oldest one."""
        view = _make_discord_view(request_id=RID)
        assert view.request_id == RID

        interaction = MagicMock()
        interaction.user.display_name = "alice"

        async def _fake_gate(_interaction, **kwargs):
            return True

        view._gate = _fake_gate
        view._finalize_embed = AsyncMock()

        captured: dict[str, Any] = {}

        def _fake_resolve(session_key, choice, *, request_id=None, **kwargs):
            captured["session_key"] = session_key
            captured["choice"] = choice
            captured["request_id"] = request_id
            return 1

        with patch("tools.approval.resolve_gateway_approval", side_effect=_fake_resolve):
            asyncio_run(view._resolve(interaction, "once", MagicMock(), "platform.discord.approval.resolved_once"))

        assert captured["session_key"] == "agent:main:discord:dm:1"
        assert captured["choice"] == "once"
        assert captured["request_id"] == RID

    def test_view_without_an_id_stays_on_the_fifo_path(self):
        """A legacy card (no forwarded id) must keep the two-argument behaviour."""
        view = _make_discord_view()
        assert getattr(view, "request_id", None) is None


def asyncio_run(coro):
    import asyncio

    return asyncio.run(coro)


@pytest.fixture
def _discord_exec_prompt():
    from gateway.platforms.base import ExecApprovalPrompt

    return ExecApprovalPrompt(
        chat_id="C1",
        session_key="agent:main:discord:dm:1",
        command="rm -rf /tmp/x",
        text="danger!",
        description="recursive delete",
        actions=[("Allow Once", "once", "green"), ("Deny", "deny", "red")],
        smart_denied=False,
        metadata={"thread_id": "t1", "approval_request_id": RID},
    )


class TestDiscordExecApprovalPromptSend:
    """The send path must hand the forwarded id to the view it builds."""

    def test_prompt_builds_a_view_bound_to_the_request(self, _discord_exec_prompt):
        dmod = _import_discord_adapter()

        adapter = object.__new__(dmod.DiscordAdapter)
        adapter._allowed_user_ids = set()
        adapter._allowed_role_ids = None
        adapter.config = SimpleNamespace(extra=None)

        built: dict[str, Any] = {}

        class _FakeChannel:
            async def send(self, **kwargs):
                return SimpleNamespace(id=42)

        def _resolve_channel(_target):
            async def _inner(*_a, **_k):
                return _FakeChannel()

            return _inner()

        async def _run():
            original_build = None

            def _capture_build(fn):
                nonlocal original_build
                return fn

            # Drive the real _send_exec_approval_prompt with _send_prompt stubbed so we can
            # capture the view the _build closure constructs.
            captured_view: dict[str, Any] = {}

            async def _fake_send_prompt(chat_id, metadata, build, **kwargs):
                send_kwargs, view = build(_FakeChannel())
                captured_view["view"] = view
                captured_view["metadata"] = metadata
                return SimpleNamespace(success=True, message_id="1")

            def _fake_send_prompt_sync(*args, **kwargs):
                # patch.object injects the instance as the first positional arg.
                return _fake_send_prompt(*args[1:], **kwargs)

            adapter._resolve_channel = _resolve_channel
            # ExecApprovalView drops buttons when a scope is unavailable; give the
            # view class a remove_item for the duration of this test only (restored
            # on exit, so no other test file observes it).
            with patch.object(dmod.ExecApprovalView, "remove_item",
                              _ensure_view_remove_item(), create=True):
                with patch.object(dmod.DiscordAdapter, "_send_prompt", _fake_send_prompt_sync):
                    await adapter._send_exec_approval_prompt(_discord_exec_prompt)

            return captured_view

        result = asyncio_run(_run())
        view = result["view"]
        assert view.session_key == "agent:main:discord:dm:1"
        assert view.request_id == RID


# ===========================================================================
# Slack
# ===========================================================================


def _ensure_slack_mock():
    """Wire up the minimal mocks required to import SlackAdapter."""
    if "slack_bolt" in sys.modules:
        return
    slack_bolt = MagicMock()
    slack_bolt.async_app.AsyncApp = MagicMock
    sys.modules["slack_bolt"] = slack_bolt
    sys.modules["slack_bolt.async_app"] = slack_bolt.async_app
    handler_mod = MagicMock()
    handler_mod.AsyncSocketModeHandler = MagicMock
    sys.modules["slack_bolt.adapter"] = MagicMock()
    sys.modules["slack_bolt.adapter.socket_mode"] = MagicMock()
    sys.modules["slack_bolt.adapter.socket_mode.async_handler"] = handler_mod
    sdk_mod = MagicMock()
    sdk_mod.web = MagicMock()
    sdk_mod.web.async_client = MagicMock()
    sdk_mod.web.async_client.AsyncWebClient = MagicMock
    sys.modules["slack_sdk"] = sdk_mod
    sys.modules["slack_sdk.web"] = sdk_mod.web
    sys.modules["slack_sdk.web.async_client"] = sdk_mod.web.async_client


_ensure_slack_mock()

from plugins.platforms.slack.adapter import SlackAdapter  # noqa: E402
from gateway.config import PlatformConfig  # noqa: E402


def _make_slack_adapter():
    config = PlatformConfig(enabled=True, token="xoxb-x")
    adapter = SlackAdapter(config)
    adapter._app = MagicMock()
    adapter._bot_user_id = "U_BOT"
    adapter._team_clients = {"T1": AsyncMock()}
    adapter._team_bot_user_ids = {"T1": "U_BOT"}
    adapter._channel_team = {"C1": "T1"}
    adapter._is_interactive_user_authorized = MagicMock(return_value=True)
    return adapter


class TestSlackApprovalRequestId:
    """The button value must carry the id, and the click must read it back out."""

    def test_button_value_embeds_the_forwarded_request_id(self):
        """An id in the card metadata must reach the button payload."""
        from gateway.platforms.base import ExecApprovalPrompt

        prompt = ExecApprovalPrompt(
            chat_id="C1",
            session_key="agent:main:slack:group:C1:1111",
            command="rm -rf /tmp/x",
            text="danger!",
            description="recursive delete",
            actions=[("Allow Once", "once", "green"), ("Deny", "deny", "red")],
            smart_denied=False,
            metadata={"approval_request_id": RID},
        )

        adapter = _make_slack_adapter()
        built: dict[str, Any] = {}

        def _fake_send_interactive_prompt(chat_id, metadata, build, label, **kwargs):
            fallback, blocks = build()
            built["blocks"] = blocks
            built["metadata"] = metadata
            return SimpleNamespace(success=True, message_id="1")

        async def _fake_send_interactive_prompt_await(*args, **kwargs):
            return _fake_send_interactive_prompt(*args, **kwargs)

        adapter._send_interactive_prompt = _fake_send_interactive_prompt_await

        asyncio_run(adapter._send_exec_approval_prompt(prompt))

        buttons = [el for b in built["blocks"] if b.get("type") == "actions" for el in b["elements"]]
        assert buttons, "the approval card must render buttons"
        values = {b["value"] for b in buttons}
        assert values == {
            f"agent:main:slack:group:C1:1111{SLACK_RID_SEP}{RID}",
        }, f"every button must carry the request id, got {values}"
        assert built["metadata"]["approval_request_id"] == RID

    def test_legacy_card_without_an_id_keeps_the_bare_session_key(self):
        """No forwarded id → the value is exactly the legacy bare session_key."""
        from gateway.platforms.base import ExecApprovalPrompt

        prompt = ExecApprovalPrompt(
            chat_id="C1",
            session_key="agent:main:slack:group:C1:1111",
            command="rm -rf /tmp/x",
            text="danger!",
            description="recursive delete",
            actions=[("Allow Once", "once", "green")],
            smart_denied=False,
            metadata={"thread_id": "t1"},
        )

        adapter = _make_slack_adapter()
        built: dict[str, Any] = {}

        def _fake_send_interactive_prompt(chat_id, metadata, build, label, **kwargs):
            _fallback, blocks = build()
            built["blocks"] = blocks

        async def _fake_send_interactive_prompt_await(*args, **kwargs):
            _fake_send_interactive_prompt(*args, **kwargs)
            return SimpleNamespace(success=True, message_id="1")

        adapter._send_interactive_prompt = _fake_send_interactive_prompt_await

        asyncio_run(adapter._send_exec_approval_prompt(prompt))

        buttons = [el for b in built["blocks"] if b.get("type") == "actions" for el in b["elements"]]
        assert {b["value"] for b in buttons} == {"agent:main:slack:group:C1:1111"}

    @pytest.mark.asyncio
    async def test_click_resolves_the_cards_own_request(self):
        """The click must resolve ITS entry — not the FIFO-oldest one."""
        adapter = _make_slack_adapter()
        client = adapter._team_clients["T1"]
        client.chat_update = AsyncMock()
        # The double-click guard pops with a default of True (already resolved);
        # a fresh pending approval must be recorded as unresolved first.
        adapter._approval_resolved["1.2"] = False

        ack = AsyncMock()
        body = {
            "message": {"ts": "1.2", "blocks": []},
            "channel": {"id": "C1"},
            "user": {"name": "alice", "id": "U_ALICE"},
        }
        action = {
            "action_id": "hermes_approve_once",
            "value": f"agent:main:slack:group:C1:1111{SLACK_RID_SEP}{RID}",
        }

        with patch("tools.approval.resolve_gateway_approval", return_value=1) as mock_resolve:
            await adapter._handle_approval_action(ack, body, action)

        mock_resolve.assert_called_once_with(
            "agent:main:slack:group:C1:1111", "once", request_id=RID)

    @pytest.mark.asyncio
    async def test_click_on_a_legacy_card_falls_back_to_fifo(self):
        """A bare session_key value keeps resolving without an id (unchanged path)."""
        adapter = _make_slack_adapter()
        client = adapter._team_clients["T1"]
        client.chat_update = AsyncMock()
        adapter._approval_resolved["1.2"] = False

        ack = AsyncMock()
        body = {
            "message": {"ts": "1.2", "blocks": []},
            "channel": {"id": "C1"},
            "user": {"name": "alice", "id": "U_ALICE"},
        }
        action = {"action_id": "hermes_approve_once", "value": "agent:main:slack:group:C1:1111"}

        with patch("tools.approval.resolve_gateway_approval", return_value=1) as mock_resolve:
            await adapter._handle_approval_action(ack, body, action)

        mock_resolve.assert_called_once_with(
            "agent:main:slack:group:C1:1111", "once", request_id=None)

    @pytest.mark.asyncio
    async def test_unauthorized_click_still_never_reaches_the_resolver(self):
        """Auth is checked before the value is ever split or resolved."""
        adapter = _make_slack_adapter()
        adapter._is_interactive_user_authorized = MagicMock(return_value=False)

        ack = AsyncMock()
        body = {
            "message": {"ts": "1.2", "blocks": []},
            "channel": {"id": "C1"},
            "user": {"name": "mallory", "id": "U_ATTACKER"},
        }
        action = {
            "action_id": "hermes_approve_once",
            "value": f"agent:main:slack:group:C1:1111{SLACK_RID_SEP}{RID}",
        }

        with patch("tools.approval.resolve_gateway_approval") as mock_resolve:
            await adapter._handle_approval_action(ack, body, action)

        mock_resolve.assert_not_called()


# ===========================================================================
# Metadata helper — shared by every adapter
# ===========================================================================


class TestMetadataRequestId:
    """``metadata_request_id`` must stay defensive (absent / blank / non-dict → None)."""

    @pytest.mark.parametrize("metadata", [None, {}, {"approval_request_id": None},
                                          {"approval_request_id": ""},
                                          {"approval_request_id": "   "},
                                          "not-a-dict", 42])
    def test_absent_or_blank_yields_none(self, metadata):
        from tools.approval import metadata_request_id

        assert metadata_request_id(metadata) is None

    def test_present_id_is_stripped(self):
        from tools.approval import metadata_request_id

        assert metadata_request_id({"approval_request_id": f"  {RID}  "}) == RID
