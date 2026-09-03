import logging

from ingestion.utils import make_robust_session

logger = logging.getLogger(__name__)

CARDS_URL = "https://api.clashroyale.com/v1/cards"


def get_card_image_urls(api_key: str, url: str = CARDS_URL) -> dict[str, str]:
    """Map every card name the API knows about to a PNG URL.

    Keys are built exactly the way ``extractors.battles.get_card_idx`` builds
    them, so they line up with the ``card_ids`` table the frontend joins against:

        base card        -> "knight"
        evolutionLevel 1 -> "evo knight"
        evolutionLevel 2 -> "hero knight"

    This has to run on the ingestion box: the Clash API whitelists the persistent
    Elastic IP that only that instance holds, so the same request from anywhere
    else (a GitHub runner, a laptop) gets a 403.

    Args:
        api_key: a Clash Royale API JWT.
        url: overridable for testing.

    Returns:
        A mapping of card name to the URL of its PNG.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    }

    session = make_robust_session()
    response = session.get(url, headers=headers)
    response.raise_for_status()
    payload = response.json()

    # supportItems holds tower troops. Including them is harmless -- they only
    # land in the output if a card of that name also exists in card_ids -- and it
    # means a card moving between the two lists does not silently lose its art.
    items = list(payload.get("items", [])) + list(payload.get("supportItems", []))

    urls: dict[str, str] = {}
    for item in items:
        name = str(item.get("name", "")).lower()
        if not name:
            continue

        icons = item.get("iconUrls") or {}
        if icons.get("medium"):
            urls[name] = icons["medium"]
        if icons.get("evolutionMedium"):
            urls[f"evo {name}"] = icons["evolutionMedium"]

        # Hero art is served from api-assets.clashroyale.com/cardheroes/, but the
        # iconUrls key for it was added after the reference snapshot was captured
        # and is not documented. Match on the key rather than guessing its name.
        for key, value in icons.items():
            if "hero" in key.lower() and value:
                urls[f"hero {name}"] = value

    logger.info(
        "fetched %d image URLs from %s (%d cards, %d evo, %d hero)",
        len(urls),
        url,
        sum(1 for k in urls if not k.startswith(("evo ", "hero "))),
        sum(1 for k in urls if k.startswith("evo ")),
        sum(1 for k in urls if k.startswith("hero ")),
    )
    return urls
