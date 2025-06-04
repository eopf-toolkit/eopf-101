import os
import requests
import re
from datetime import datetime, timedelta

HARBOR_URL = os.environ["OVH_HARBOR_REGISTRY"]
PROJECT = "eopf-toolkit-dev"
REPO = "eopf-toolkit-dev"
USERNAME = os.environ["OVH_HARBOR_ROBOT_USERNAME"]
PASSWORD = os.environ["OVH_HARBOR_ROBOT_PASSWORD"]
DAYS_OLD = int(os.environ["IMAGE_CLEANUP_DAYS"])
TARGET_REGEX = re.compile(r"^(pr-.*|release-.*)$")
CUTOFF = datetime.now() - timedelta(days=DAYS_OLD)


url = f"{HARBOR_URL}/api/v2.0/projects/{PROJECT}/repositories/{REPO}/artifacts?page_size=100"
resp = requests.get(url, auth=(USERNAME, PASSWORD), headers={
    "Accept": "application/json",
    "Content-Type": "application/json",
})
resp.raise_for_status()

artifacts = resp.json()

for artifact in artifacts:
    tags = artifact.get("tags", [])
    if not tags:
        continue

    tag_names = {tag["name"] for tag in tags}
    if "latest" in tag_names:
        print(f"Skipping artifact {artifact['digest']} — has 'latest'")
        continue

    for tag in tags:
        name = tag["name"]
        if not TARGET_REGEX.match(name):
            continue

        pushed = datetime.strptime(tag["push_time"], "%Y-%m-%dT%H:%M:%S.%fZ")
        if pushed >= CUTOFF:
            continue

        digest = artifact["digest"]
        print(f"Deleting artifact {digest} due to tag '{name}' (pushed {pushed})")
        del_url = f"{HARBOR_URL}/api/v2.0/projects/{PROJECT}/repositories/{REPO}/artifacts/{digest}"
        del_resp = requests.delete(
            del_url, 
            auth=(USERNAME, PASSWORD),
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json"
            }
        )
        if del_resp.status_code == 200:
            print(f"Deleted {name}")
        else:
            print(f"Failed to delete {name}: {del_resp.status_code} {del_resp.text}")
        break
