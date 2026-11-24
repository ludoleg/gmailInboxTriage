"""Slack Block Kit morning digest formatting and webhook delivery."""

from inbox_triage.slack.client import SlackWebhookClient
from inbox_triage.slack.formatter import build_slack_payloads

__all__ = ["SlackWebhookClient", "build_slack_payloads"]
