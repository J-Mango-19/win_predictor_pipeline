import logging
import psycopg
import requests
from pathlib import Path
from datetime import datetime, timedelta, timezone
from services.ingestion.src.ingestion.utils import make_robust_session, setup_process_file_logger, construct_db_URI


def fetch_and_store_players(session: requests.sessions.Session, clan_id: str, api_key: str, logger: logging.Logger, weeks_since_last_game: int=4) -> list:
    """Fetches the list of members for a given clan ID who were recently active """

    clean_tag = clan_id.strip("#").upper()
    url = f"https://api.clashroyale.com/v1/clans/%23{clean_tag}/members"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    }

    response = session.get(url, headers=headers)  
    response.raise_for_status()  # Raises an error for bad responses
    members = response.json().get("items", [])  

    # Filtering 
    active_member_ids = []
    
    # Define the 6-week cutoff point relative to the current UTC time
    cutoff_time = datetime.now(timezone.utc) - timedelta(weeks=weeks_since_last_game)
    
    for member in members:
        last_seen_str = member.get("lastSeen")
        if not last_seen_str:
            logger.info("skipping a clan member, no lastSeen date given")
            continue
            
        try:
            # Parse the API string format: "20260611T030424.000Z"
            # .replace(tzinfo=timezone.utc) makes it safe to compare with modern UTC datetimes
            last_seen_dt = datetime.strptime(last_seen_str, "%Y%m%dT%H%M%S.%fZ").replace(tzinfo=timezone.utc)
            
            # Check if the player was seen within the 6-week window
            if last_seen_dt >= cutoff_time:
                active_member_ids.append(member["tag"])
                
        except ValueError:
            # Safeguard against unexpected or malformed date strings
            logger.info("skipping a clan member, could not parse their lastSeen date")
            continue

    return active_member_ids

def store_clan_members(max_source_clans: int, api_key: str, log_dir: Path):
    """ Stores the member IDs of clans into active_players. """

    logger = setup_process_file_logger(log_dir)

    with psycopg.connect(construct_db_URI()) as conn:
        with conn.cursor() as cur:
            # Claim up to max_clans using FOR UPDATE SKIP LOCKED
            cur.execute("""
                UPDATE clans
                SET claimed = true
                WHERE clan_id IN (
                    SELECT clan_id
                    FROM clans
                    WHERE claimed = false AND members_collected = false
                    LIMIT %s
                    FOR UPDATE SKIP LOCKED
                )
                RETURNING clan_id;
            """, (max_source_clans,))

            conn.commit()

            # Extract the claimed clan_ids into a flat list
            claimed_rows = cur.fetchall()

    clan_ids = [row[0] for row in claimed_rows]
    logger.info(f"{len(clan_ids)} clan IDs to be processed (maximum {max_source_clans})")
    
    if not clan_ids:
        logger.warning("No valid/uncollected clan IDs found in database! exiting store_clan_members().")
        return  
    
    # Gather all members of these clans
    session = make_robust_session()

    member_ids = []
    n_clans = len(clan_ids)
    for i, clan_id in enumerate(clan_ids):
        members = get_active_clan_members(session, clan_id, api_key, logger=logger)
        if members:
            member_ids.extend(members)
        else:
            logger.info("clan found w/o any active members")
        if i % int(n_clans / 5) == 0:
            logger.info(f"{i+1:03d}/{n_clans} clans' members processed. {len(member_ids)} members found so far")
    session.close()

    with psycopg.connect(construct_db_URI()) as conn:
        with conn.cursor() as cur:
            # Insert those member_IDs into active_players
            if member_ids:
                # Format data for executemany
                records = [(m_id,) for m_id in member_ids]
                
                cur.executemany("""
                    INSERT INTO active_players (player_id, claimed)
                    VALUES (%s, false)
                    ON CONFLICT (player_id) DO NOTHING;
                """, records)
                
            # Mark clans as collected and release the claim lock
            cur.execute("""
                UPDATE clans
                SET members_collected = true, claimed = false
                WHERE clan_id = ANY(%s);
            """, (clan_ids,))

            conn.commit()

    logger.info("clan members inserted into active_players successfully.")