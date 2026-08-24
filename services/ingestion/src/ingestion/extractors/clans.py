import logging
import requests
import psycopg
import random
from pathlib import Path
from ingestion.utils import make_robust_session, setup_process_file_logger, construct_db_URI

def get_region_IDs(cr_api_key: str, url: str="https://api.clashroyale.com/v1/locations") -> set:
    """ Returns a set of all valid region IDs in the CR API, except CN (different version of the game) """
    
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {cr_api_key}"
    }

    session = make_robust_session()
    try:
        response = session.get(url, headers=headers)
        response.raise_for_status()  # check for HTTP errors
    except requests.exceptions.HTTPError as err:
        logging.error(f"An error occurred fetching region IDs: {err}")
        raise

    locations_dict = response.json()
    item_list = locations_dict.get('items', [])
    location_IDs = set()
    for item in item_list:
        if item.get('name') == 'China':
            continue
        location_IDs.add(item.get('id', ''))
    
    if '' in location_IDs:
        location_IDs.remove('')  # can't lookup empty strings!
    
    return location_IDs


def fetch_and_store_clans(new_clan_limit: int, cr_api_key: str, valid_loc_IDs: list, log_dir: Path, batch_size: int=500) -> None:
    """ writes up to new_clan_limit + batch_size clan IDs to the database """

    logger = setup_process_file_logger(log_dir)

    def generate_random_clan_request():
        loc = random.choice(valid_loc_IDs)
        min_members = random.randint(2, 5) # >= 2
        min_score = random.randint(1, 1000) # >= 1
        return f"https://api.clashroyale.com/v1/clans?locationId={loc}&minMembers={min_members}&minScore={min_score}&limit={batch_size}", loc

    # load known clans from the database to avoid re-collection
    with psycopg.connect(construct_db_URI()) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT clan_id
                FROM clans;
            """)
            known_clans = {row[0] for row in cur.fetchall()}
    
    # setup network connection
    session = make_robust_session()
    headers = {
        "Authorization": f"Bearer {cr_api_key}",
        "Accept": "application/json",    
    }  

    iters = 0
    new_clans = set()

    while(len(new_clans) < new_clan_limit):
        # If we've exhausted all locations, break to avoid an infinite loop
        if not valid_loc_IDs:  
            logger.info("Exhausted all location IDs without reaching the desired number of new clans.")
            break
        
        url, loc = generate_random_clan_request()
        try:
            response = session.get(url, headers=headers)
            response.raise_for_status()
        except Exception as http_err:
            logger.info(f"HTTP error ocurred: {http_err}")
            continue

        cur_clans = response.json().get("items", [])  # yields a list of clans
        cur_clans = set(clan['tag'] for clan in cur_clans)  # extract clan tags
        cur_clans = cur_clans.difference(known_clans) # remove old already used clans
        if len(cur_clans) < 5 and loc in valid_loc_IDs:
            valid_loc_IDs.remove(loc) # remove locations that are out of clans
        iters += 1
        new_clans.update(cur_clans)
        known_clans.update(cur_clans)
        logger.info(f"{iters=:02d}, {len(cur_clans)} clans found this iteration, {len(new_clans)} new clans found, {len(valid_loc_IDs)} locations remaining for search.")

    session.close()
    new_clans = list(new_clans)
    
    # write new_clans to the database
    with psycopg.connect(construct_db_URI()) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO clans (clan_id, members_collected, claimed)
                SELECT
                    unnest(%s::text[]),
                    FALSE,
                    FALSE
                ON CONFLICT (clan_id) DO NOTHING;
            """, (new_clans,))
    
    logger.info(f"{len(new_clans)} fresh clans written to database")