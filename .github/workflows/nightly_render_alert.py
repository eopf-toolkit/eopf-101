import logging
import os
from requests.sessions import Session

logger = logging.getLogger()

def post_alert_to_slack(session: Session, slack_channel: str, run_url: str) -> None:
    msg = f"""<!here> - Nightly Render Failed :night_with_stars: :sob:
Access the build here: {run_url} to see the errors
It's likely the next `Publish` will fail unless these are fixed.
"""
    resp = session.post(
        "https://slack.com/api/chat.postMessage",
        json={"text": msg, "channel": slack_channel},
    )
    try:
        resp.raise_for_status()
    except Exception:
        # These aren't critical, so we can just log the error
        logger.exception(f"Couldn't send slackbot message to {slack_channel}")


if __name__ == "__main__":
    token = os.environ["SLACKBOT_TOKEN"]
    slack_channel = os.environ["SLACKBOT_CHANNEL"]
    run_url = os.environ["RUN_URL"]
    session = Session()
    session.headers.update({"Authorization": f"Bearer {token}"})
    post_alert_to_slack(session=session, slack_channel=slack_channel, run_url=run_url)
