import gc
import time
import psycopg
import requests
from pathlib import Path
from datetime import datetime, timezone, timedelta
from ingestion.utils import make_robust_session, setup_process_file_logger, construct_db_URI, translate_gameModes_to_apiNames
from common.constants import INVALID_TOKEN, GAME_COLS
from common.utils import ensure_utc, parse_api_timestamp


def get_battle_log(player_tag: str, session: requests.sessions.Session, api_key: str, logger) -> list:
    """Fetches the battle log for a specific Clash Royale player tag."""

    # Format the tag by removing the '#' if present and URL-encoding it
    clean_tag = player_tag.strip("#").upper()
    url = f"https://api.clashroyale.com/v1/players/%23{clean_tag}/battlelog"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    }
    try:
        response = session.get(url, headers=headers)
        response.raise_for_status()
    except requests.exceptions.HTTPError as err:
        if logger is not None:
            logger.info(f"Unable to download a battle log; HTTP Error occurred: {err}")
        return []

    return response.json()

def get_card_lvl(card: dict) -> int:
    """Convert levels as given by the CR API to levels as seen in-game.

    Gameplay logic places all cards on a level range from 1-16, but the API
    returns levels in a range based on the card's type (eg, common, rare, etc).
    For example, epic cards range from level 1-11 in the API but should be 6-16 in-game.

    Args:
        card: dictionary representing a card as returned by the CR API

    Returns:
        The card level as seen in-game.
    """
    api_lvl = int(card['level']) 
    api_max_lvl = int(card['maxLevel'])

    return api_lvl + (16 - api_max_lvl)


def filter_by_time(battles: list, time_boundary: datetime) -> list:
    """ removes battles older than the time_boundary

    Args:
        battles: a list of game dictionaries, eg the result of downloading a player's battleLog
        time_boundary: games older than this time are filtered out

    Returns:
        a list of battles that are more recent than time_boundary
    """
    time_boundary = ensure_utc(time_boundary)
    recent_battles = []
    for battle in battles:
        if parse_api_timestamp(battle['battleTime']) > time_boundary:
            recent_battles.append(battle)
    
    return recent_battles

def filter_by_game_mode(battles: list, allowed_gameModes: list) -> list:
    """ removes battles that are not of the allowed game modes

    Args:
        battles: a list of game dictionaries, eg the result of downloading a player's battleLog
        allowed_gameModes: a list of allowed battle game modes (eg, ["draft"])

    Returns:
        a list of battles that are of the allowed game modes
    """
    allowed_gameModes = [item.lower() for item in allowed_gameModes]
    filtered_battles = []
    for battle in battles:
        if battle['gameMode']['name'].lower() in allowed_gameModes and battle['type'].lower() != "trail":
            filtered_battles.append(battle)
    
    return filtered_battles

def get_card_idx(card: dict, name_to_idx: dict, logger) -> int:
    """ Returns the index of a card based on its name and evolution level.

    Pulls the idx from name_to_idx if it's in there.
    Otherwise:
        - it pulls the card idx from the database (making a new entry if necessary)
        - it updates name_to_idx (to minimize future database queries)

    Atomic queries ensure this fxn is safe to run concurrently

    Args: 
        card: a dictionary representing a card as returned by the CR API
        name_to_idx: a dictionary mapping card names to their corresponding indices
        logger: a logging.Logger object for logging information
    
    Returns:
        a card idx
    """
    name = card['name'].lower()

    evo_lvl = card.get('evolutionLevel', None)
    if evo_lvl is not None:
        if evo_lvl == 1:
            name = f"evo {name}"
        elif evo_lvl == 2:
            name = f"hero {name}"
        else:
            name = f"evo hero {name}"
            logger.warning(f"Found a card with {evo_lvl=}. Treating it as 'evo hero {name}' but no idx exists for this")

    if name in name_to_idx:
        return name_to_idx[name]

    logger.warning(f"Card '{name}' not in local dict... synchronizing with DB.")

    # This inserts the card if it doesn't exist, or safely fetches the existing ID if it does.
    get_or_create_sql = """
    WITH ins AS (
        INSERT INTO card_ids (name)
        VALUES (%s)
        ON CONFLICT (name) DO NOTHING
        RETURNING id
    )
    SELECT id FROM ins
    UNION ALL
    SELECT id FROM card_ids WHERE name = %s
    LIMIT 1;
    """

    try:
        with psycopg.connect(construct_db_URI()) as conn:
            with conn.cursor() as cur:
                cur.execute(get_or_create_sql, (name, name))
                result = cur.fetchone()
                
                if result:
                    idx = result[0]
                    # Update local process dictionary
                    name_to_idx[name] = idx
                    return idx
                else:
                    raise RuntimeError(f"Failed to retrieve or generate an ID for {name}")
                    
    except Exception as e:
        logger.error(f"Database error while processing card '{name}': {e}")
        return INVALID_TOKEN


def format_battle(battle: dict, name_to_idx: dict, logger) -> tuple:
    """ 
    Formats a battle dictionary as a list containing
        - winner ID
        - loser ID
        - time of battle
        - gameMode
        - winner card types
        - loser card types
        - winner card lvls
        - loser card lvls
    In the event of a tie, the function returns (None, opponent_id) and does not record the battle in the DB.

    Args:
        battle: a dictionary representing a battle as returned by the CR API
        name_to_idx: a dictionary mapping card names to their corresponding indices
        logger: a logging.Logger object for logging information

    Returns:
      (formatted battle, opponent_id)
    """

    # winner
    if battle['team'][0]['crowns'] > battle['opponent'][0]['crowns']:
        winner = 'team'
        loser = 'opponent'
    elif battle['team'][0]['crowns'] < battle['opponent'][0]['crowns']:
        winner = 'opponent'
        loser = 'team'
    else:
        # not recording ties
        return None, battle['opponent'][0]['tag']
    
    out = []
    out.append(battle[winner][0]['tag'])
    out.append(battle[loser][0]['tag'])
    out.append(battle['battleTime'])
    
    # game mode
    out.append(battle['gameMode']['name'])

    # winner cards idx
    for card in battle[winner][0]['cards']:
        out.append(get_card_idx(card, name_to_idx, logger))
    # loser cards idx
    for card in battle[loser][0]['cards']:
        out.append(get_card_idx(card, name_to_idx, logger))

    # winner cards level
    for card in battle[winner][0]['cards']:
        out.append(get_card_lvl(card))
    # loser cards level
    for card in battle[loser][0]['cards']:
        out.append(get_card_lvl(card))
    
    if len(out) != 36:
        logger.warning(f"Battle formatting error: expected 36 fields, got {len(out)}. Battle dict: {battle}")
        logger.info("Not recording this battle in DB")
        return None, battle['opponent'][0]['tag']

    return out, battle['opponent'][0]['tag']


def is_recently_active(battles: list, weeks: int) -> bool:
    """ Returns a boolean indicating whether the most recent battle is less than `weeks` old.

    Args:
        battles: a player's battle log; a list of battle dicts
        weeks: battles older than this number of weeks don't contribute to a player being active

    Returns:
        a boolean, whether the player corresponding to this list of battles is active
    """
    if len(battles) == 0:
        return False # no battles logged, so player not active

    cutoff_time = datetime.now(timezone.utc) - timedelta(weeks=weeks)
    most_recent_battle_time = parse_api_timestamp(battles[0]['battleTime'])

    return most_recent_battle_time > cutoff_time

def claim_player_ids(max_player_ids: int, logger) -> list:
    """ returns a list of player ID's from the active_players table

    Concurrent processes calling this function will receive disjoint sets of player ID's

    Args:
        max_player_ids: the maximum number of player ID's to claim from the active_players table
        logger: a logging.Logger object for logging information
    
    Returns:
        A list of player ID's claimed from the active_players table.
    """

    claim_players_sql = """
        WITH chosen AS (
            SELECT ap.player_id
            FROM active_players ap
            WHERE COALESCE(ap.claimed, false) = false
            LIMIT %s
            FOR UPDATE OF ap SKIP LOCKED
        )
        UPDATE active_players ap
        SET claimed = true
        FROM chosen
        WHERE ap.player_id = chosen.player_id
        RETURNING ap.player_id;
    """
    s = time.time()
    with psycopg.connect(construct_db_URI()) as conn:
        with conn.cursor() as cur:
            cur.execute(claim_players_sql, (max_player_ids,))
            claimed_player_ids = [row[0] for row in cur.fetchall()]
    e = time.time()
    logger.info(f"gathering {len(claimed_player_ids)} IDs from the db took: {e-s:.02f}s")

    if not claimed_player_ids:
        logger.warning("Failed to claim player_id's from active_players")

    return claimed_player_ids


def fetch_and_store_games(api_key: str, oldest_time_allowed: datetime, game_modes_allowed: list, inactivity_limit_wks: int, card_to_idx: dict, log_dir: Path, soft_games_limit: int, write_player_chunk_size: int=500, read_player_chunk_size: int=25_000) -> None:
    """
    While there are unclaimed players in the active_players table... 
        - claims a chunk of players from the active_players table
        - downloads, filters, and formats the games from their battle logs
        - writes the formatted games to the games table
        - writes opponents to active_players
    
    Args:
        api_key_path: path of the text file holding the Supercell API key
        oldest_time_allwed: games older than this time will filtered out, ie not written to the games table
        game modes allowed: games not falling into one of these categories will be filtered out, ie not written to the games table
        inactivity_limit_wks: players whose most recent battle is older than this number of weeks will be removed from active_players
        card_to_idx: mapping from card name to numeric index
        write_player_chunk_size: the number of players' games we write to the database at once
        read_player_chunk_size: the number of players we claim from the database at once
    
        read/write player chunk size are different bc 
            - it's most efficient to read player ids from the database in large chunks
            - it's most efficient to write games (corresponding to a number of players) in small chunks

    Returns:
        None
    """

    logger = setup_process_file_logger(log_dir)
    api_game_modes_to_collect = translate_gameModes_to_apiNames(game_modes_allowed)

    def download_chunk_battles(player_id_chunk: list) -> tuple:
        """ returns a list of formatted battles given a list of player IDs """
        opponents = set()
        formatted_battles = []
        stale_player_ids = [] # players we will remove from active_players table (as they are no longer active)
        for player_id in player_id_chunk:
            battles = get_battle_log(player_id, session, api_key, logger) # returns a list of dicts (each dict is a battle)
            new_battles = filter_by_time(battles, oldest_time_allowed)
            valid_battles = filter_by_game_mode(new_battles, api_game_modes_to_collect)
            for b in valid_battles:
                formatted_battle, opp_id = format_battle(b, card_to_idx, logger)
                opponents.add(opp_id)
                if formatted_battle is not None:
                    # None in the event of a tie
                    formatted_battles.append(formatted_battle)
            
            # slate inactive players for removal from active_players
            if not is_recently_active(battles, inactivity_limit_wks):
                stale_player_ids.append(player_id)

        return formatted_battles, opponents, stale_player_ids
    
    def write_chunk_battles(battle_chunk: list, tries: int = 0) -> bool:
        """ 
        writes a list of formatted battles to the games table
        returns true on success, false on failure
        """
        if not battle_chunk:
            logger.warning("empty list of battles passed to write_chunk_battles()")
            return False

        if tries > 3:
            logger.warning("Failed to write a chunk of battles to the database after 3 tries")
            logger.warning("Exiting store_games()")
            return False
        
        try:
            insert_games_sql = f"""
                INSERT INTO games ({", ".join(GAME_COLS)})
                VALUES ({", ".join(["%s"] * len(GAME_COLS))})
                ON CONFLICT DO NOTHING;
            """
            
            s = time.time()
            with psycopg.connect(construct_db_URI()) as conn:
                with conn.cursor() as cur:
                    cur.executemany(insert_games_sql, battle_chunk)
                    battles_inserted = cur.rowcount
            
            e = time.time()
            
            # Handle cases where rowcount might be negative or undefined due to pipeline errors
            battles_inserted = max(0, battles_inserted) 
            pct_new = battles_inserted / len(battle_chunk)
            
            nonlocal total_battles_inserted
            total_battles_inserted += battles_inserted
            logger.info(f"  Wrote a chunk of {battles_inserted} battles to the DB. {round(pct_new*100, 0)}% were new; took {(e-s)/60:.02f} mins")
            
            return True # Successfully finished this branch!

        except Exception as db_err:
            logger.error(f"Database error encountered: {db_err}", exc_info=True)
            
            tries += 1
            logger.info(f"Retrying write_chunk_battles... (Attempt {tries}/3)")
            
            return write_chunk_battles(battle_chunk, tries)

    
    def write_opponents(opponent_pids: set):
        """ writes a chunk of player ids to active_players """
        insert_pids_sql = f"""
            INSERT INTO active_players (player_id, claimed)
            VALUES (%s, true)
            ON CONFLICT (player_id) DO NOTHING;
        """
        opponents = [(o,) for o in sorted(opponent_pids)]
        with psycopg.connect(construct_db_URI()) as conn:
            with conn.cursor() as cur:
                cur.executemany(insert_pids_sql, opponents)
    
    def remove_ids(ids: list):
        """ deletes a set of player ids from active_players """
        delete_pids_sql = """
            DELETE FROM active_players 
            WHERE player_id = %s;
        """
        ids_tuples = [(id,) for id in sorted(ids)]
        with psycopg.connect(construct_db_URI()) as conn:
            with conn.cursor() as cur:
                cur.executemany(delete_pids_sql, ids_tuples)
        nonlocal stale_players_removed
        stale_players_removed += len(ids_tuples)
    
    def count_games_in_db():
        """ returns the number of games stored in the games table """
        count_games_sql = """
            SELECT COUNT(*)
            FROM games;
        """
        with psycopg.connect(construct_db_URI()) as conn:
            with conn.cursor() as cur:
                cur.execute(count_games_sql) 
                count = cur.fetchone()[0]
        return count


    total_battles_found = 0
    total_battles_inserted = 0
    stale_players_removed = 0

    session = make_robust_session()
    num_games_collected = 0

    while (claimed_player_ids := claim_player_ids(max_player_ids=read_player_chunk_size, logger=logger)) and num_games_collected < soft_games_limit:
        # claimed_player_ids is a list of player ids, with length at most read_player_chunk_size 

        # download battles & write them in chunks
        for start_idx in range(0, len(claimed_player_ids), write_player_chunk_size):
            # make player id chunk
            end_idx = min(start_idx + write_player_chunk_size, len(claimed_player_ids))
            player_id_chunk = claimed_player_ids[start_idx : end_idx]

            # download battles
            formatted_battles, new_opponents, stale_player_ids = download_chunk_battles(player_id_chunk)
            total_battles_found += len(formatted_battles)

            # write opponents
            write_opponents(new_opponents)

            # remove stale player ids
            remove_ids(stale_player_ids)

            # write battles to database
            success = write_chunk_battles(formatted_battles)
            if not success:
                logger.warning("failed to write a battle chunk to the DB")
                return
            
            del formatted_battles
            del new_opponents
            del stale_player_ids
            gc.collect()

        logger.info("Processed a read chunk of players. Continuing to next chunk...")
        num_games_collected = count_games_in_db()

    logger.info("store_games ran out of non-claimed player Ids, returned successfully.")
    logger.info(
    f"Battles found: {total_battles_found} | "
    f"Battles inserted: {total_battles_inserted} | "
    f"Inactive Players removed: {stale_players_removed}"
    )