from pathlib import Path

CONSTANTS_FILE_PATH = Path(__file__).resolve()
PROJECT_ROOT = CONSTANTS_FILE_PATH.parents[4]
print(PROJECT_ROOT)


GAME_COLS = [
    "winner_id", "loser_id", "time", "game_mode",
    "winner_card_0", "winner_card_1", "winner_card_2", "winner_card_3",
    "winner_card_4", "winner_card_5", "winner_card_6", "winner_card_7",
    "loser_card_0", "loser_card_1", "loser_card_2", "loser_card_3",
    "loser_card_4", "loser_card_5", "loser_card_6", "loser_card_7",
    "winner_card_0_level", "winner_card_1_level", "winner_card_2_level", "winner_card_3_level",
    "winner_card_4_level", "winner_card_5_level", "winner_card_6_level", "winner_card_7_level",
    "loser_card_0_level", "loser_card_1_level", "loser_card_2_level", "loser_card_3_level",
    "loser_card_4_level", "loser_card_5_level", "loser_card_6_level", "loser_card_7_level",
]

WINNER_CARD_COLS = [f'winner_card_{i}' for i in range(8)]
LOSER_CARD_COLS  = [f'loser_card_{i}' for i in range(8)]
WINNER_LVL_COLS  = [f'winner_card_{i}_level' for i in range(8)]
LOSER_LVL_COLS   = [f'loser_card_{i}_level' for i in range(8)]

# Used to represent an invalid card ID
INVALID_TOKEN = 254


# Mapping of game modes to their corresponding API identifiers
GAME_MODES = {
  "draft": "DraftMode",
  "triple draft": "Draft_Competitive",
  "mega draft": "PickMode",
  "ladder": "Ladder",
  "ranked ladder": "Ranked1v1_NewArena"
}