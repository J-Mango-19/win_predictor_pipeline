import { useEffect, useMemo, useState } from "react";
import * as ort from "onnxruntime-web/wasm";

const DECK_SIZE = 8;
const DEFAULT_LEVEL = 11;
// Must match MAX_CARD_LEVEL in services/training/src/training/data.py, which is
// what the levels were divided by when the model was trained.
const MAX_CARD_LEVEL = 16;

const assetPath = (path: string) => `${import.meta.env.BASE_URL}${path}`;
// Everything else is named by the manifest, which prepare-assets.mjs writes at
// build time with content-addressed filenames.
const MANIFEST_PATH = assetPath("model-manifest.json");
const LEVELS = Array.from({ length: MAX_CARD_LEVEL }, (_, index) => index + 1);

// Graph I/O of the pipeline's model. Defined in
// services/training/src/training/quantize.py (INPUT_NAMES / OUTPUT_NAME);
// changing either side means changing both.
const INPUT_DECK_A = "deck_a";
const INPUT_DECK_B = "deck_b";
const INPUT_LVLS_A = "deck_a_lvls";
const INPUT_LVLS_B = "deck_b_lvls";
const OUTPUT_LOGIT = "logit";

type CardMap = Record<string, number>;
type PngUrlMap = Record<string, string>;

type Manifest = {
  schemaVersion: number;
  model: string;
  cardMap: string;
  imageUrls: string;
  /** Rows in the model's embedding table. null for a model published before the
   *  training stage emitted a metadata sidecar. */
  vocabSize: number | null;
  trainedAt: string | null;
  trainingCommit: string | null;
};

type Card = {
  name: string;
  tokenId: number;
  displayName: string;
  imageUrl: string;
};

type DeckSlot = {
  cardName: string | null;
  level: number;
};

type PlayerKey = "playerOne" | "playerTwo";

type PredictionState = {
  status: "idle" | "loading-model" | "running" | "complete" | "error";
  probability: number | null;
  message: string;
};

const emptyDeck = (): DeckSlot[] =>
  Array.from({ length: DECK_SIZE }, () => ({ cardName: null, level: DEFAULT_LEVEL }));

const cardDisplayName = (name: string) => name.replaceAll("_", " ");

const imageKey = (name: string) => cardDisplayName(name).toLowerCase();

const toCards = (
  cardMap: CardMap,
  pngUrls: PngUrlMap,
  vocabSize: number | null,
): { cards: Card[]; hidden: number } => {
  const all = Object.entries(cardMap);

  // The embedding table has `vocabSize` rows, but card_ids legitimately holds
  // cards that never appeared in an exported game -- their ids sit past the end
  // of the table. ORT's WASM backend reads out of bounds rather than raising, so
  // a bad token yields a plausible-looking wrong answer. Drop them here.
  const usable =
    vocabSize === null ? all : all.filter(([, id]) => id >= 0 && id < vocabSize);

  const cards = usable
    .map(([name, tokenId]) => ({
      name,
      tokenId,
      displayName: cardDisplayName(name),
      imageUrl: pngUrls[imageKey(name)] ?? "",
    }))
    .sort((a, b) => a.tokenId - b.tokenId);

  return { cards, hidden: all.length - usable.length };
};

const selectedNames = (deck: DeckSlot[]) =>
  new Set(deck.map((slot) => slot.cardName).filter(Boolean) as string[]);

const sigmoid = (logit: number) =>
  logit >= 0 ? 1 / (1 + Math.exp(-logit)) : Math.exp(logit) / (1 + Math.exp(logit));

const getInitials = (name: string) =>
  cardDisplayName(name)
    .split(" ")
    .map((part) => part[0])
    .join("")
    .slice(0, 3)
    .toUpperCase();

function App() {
  const [cards, setCards] = useState<Card[]>([]);
  const [cardLoadError, setCardLoadError] = useState("");
  const [playerOne, setPlayerOne] = useState<DeckSlot[]>(emptyDeck);
  const [playerTwo, setPlayerTwo] = useState<DeckSlot[]>(emptyDeck);
  const [activeSlot, setActiveSlot] = useState<{ player: PlayerKey; index: number } | null>(null);
  const [session, setSession] = useState<ort.InferenceSession | null>(null);
  const [manifest, setManifest] = useState<Manifest | null>(null);
  const [hiddenCardCount, setHiddenCardCount] = useState(0);
  const [prediction, setPrediction] = useState<PredictionState>({
    status: "idle",
    probability: null,
    message: "Choose 16 cards to compare the decks.",
  });

  useEffect(() => {
    let cancelled = false;

    const fetchJson = async <T,>(url: string, label: string): Promise<T> => {
      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(`${label} failed to load: ${response.status}`);
      }
      return (await response.json()) as T;
    };

    const load = async () => {
      // The manifest is the one asset without a content-addressed name, so it
      // is the only one that can go stale. "no-cache" forces a conditional
      // revalidation (a ~600 byte 304), rather than "no-store" which would
      // throw away the 304 too.
      const response = await fetch(MANIFEST_PATH, { cache: "no-cache" });
      if (!response.ok) {
        throw new Error(
          `model-manifest.json missing (${response.status}). ` +
            "Run `npm run fetch-assets` to download the model from S3.",
        );
      }
      const loaded = (await response.json()) as Manifest;

      const [cardMap, pngUrls] = await Promise.all([
        fetchJson<CardMap>(assetPath(loaded.cardMap), "Card list"),
        fetchJson<PngUrlMap>(assetPath(loaded.imageUrls), "Card image URLs"),
      ]);
      if (cancelled) {
        return;
      }

      const { cards: usable, hidden } = toCards(cardMap, pngUrls, loaded.vocabSize);
      setManifest(loaded);
      setCards(usable);
      setHiddenCardCount(hidden);
    };

    load().catch((error: unknown) => {
      if (!cancelled) {
        setCardLoadError(error instanceof Error ? error.message : "Card list failed to load.");
      }
    });

    return () => {
      cancelled = true;
    };
  }, []);

  const playerOneSelected = useMemo(() => selectedNames(playerOne), [playerOne]);
  const playerTwoSelected = useMemo(() => selectedNames(playerTwo), [playerTwo]);
  const isComplete = [...playerOne, ...playerTwo].every((slot) => slot.cardName);
  const cardByName = useMemo(() => new Map(cards.map((card) => [card.name, card])), [cards]);

  const updateSlotCard = (player: PlayerKey, index: number, cardName: string) => {
    const updater = (deck: DeckSlot[]) =>
      deck.map((slot, slotIndex) =>
        slotIndex === index ? { ...slot, cardName, level: slot.level || DEFAULT_LEVEL } : slot,
      );

    if (player === "playerOne") {
      setPlayerOne(updater);
    } else {
      setPlayerTwo(updater);
    }

    setActiveSlot(null);
    setPrediction((current) => ({
      ...current,
      status: current.status === "complete" ? "idle" : current.status,
      message:
        current.status === "complete"
          ? "Deck changed. Run prediction again when ready."
          : current.message,
    }));
  };

  const updateSlotLevel = (player: PlayerKey, index: number, level: number) => {
    const updater = (deck: DeckSlot[]) =>
      deck.map((slot, slotIndex) => (slotIndex === index ? { ...slot, level } : slot));

    if (player === "playerOne") {
      setPlayerOne(updater);
    } else {
      setPlayerTwo(updater);
    }

    setPrediction((current) => ({
      ...current,
      status: current.status === "complete" ? "idle" : current.status,
      message:
        current.status === "complete"
          ? "Level changed. Run prediction again when ready."
          : current.message,
    }));
  };

  const clearSlot = (player: PlayerKey, index: number) => {
    const updater = (deck: DeckSlot[]) =>
      deck.map((slot, slotIndex) =>
        slotIndex === index ? { cardName: null, level: DEFAULT_LEVEL } : slot,
      );

    if (player === "playerOne") {
      setPlayerOne(updater);
    } else {
      setPlayerTwo(updater);
    }
    setPrediction({
      status: "idle",
      probability: null,
      message: "Choose 16 cards to compare the decks.",
    });
  };

  const loadSession = async () => {
    if (session) {
      return session;
    }
    if (!manifest) {
      throw new Error("Model manifest has not loaded yet.");
    }

    setPrediction({
      status: "loading-model",
      probability: null,
      message: "Loading the ONNX model in your browser...",
    });

    // No externalData: the pipeline quantizes to a single self-contained file
    // (services/training/src/training/quantize.py calls plain onnx.save). That
    // is also what lets prepare-assets.mjs rename it by content hash.
    const nextSession = await ort.InferenceSession.create(assetPath(manifest.model), {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });
    setSession(nextSession);
    return nextSession;
  };

  const runPrediction = async () => {
    if (!isComplete) {
      return;
    }

    try {
      const currentSession = await loadSession();

      setPrediction({
        status: "running",
        probability: null,
        message: "Running deck matchup inference...",
      });

      const vocabSize = manifest?.vocabSize ?? null;
      const tokenIds = (deck: DeckSlot[]) =>
        deck.map((slot) => {
          const card = slot.cardName ? cardByName.get(slot.cardName) : null;
          if (!card) {
            throw new Error("A selected card is missing from the token map.");
          }
          // toCards already filters these out of the picker; this catches any
          // other path into a deck (a future import feature, say) before the
          // out-of-range read reaches ORT, which would not report it.
          if (vocabSize !== null && (card.tokenId < 0 || card.tokenId >= vocabSize)) {
            throw new Error(
              `"${card.displayName}" is outside this model's vocabulary (${vocabSize} cards).`,
            );
          }
          return BigInt(card.tokenId);
        });

      const levels = (deck: DeckSlot[]) => deck.map((slot) => slot.level / MAX_CARD_LEVEL);

      const feeds: Record<string, ort.Tensor> = {
        [INPUT_DECK_A]: new ort.Tensor(
          "int64",
          BigInt64Array.from(tokenIds(playerOne)),
          [1, DECK_SIZE],
        ),
        [INPUT_DECK_B]: new ort.Tensor(
          "int64",
          BigInt64Array.from(tokenIds(playerTwo)),
          [1, DECK_SIZE],
        ),
        [INPUT_LVLS_A]: new ort.Tensor(
          "float32",
          Float32Array.from(levels(playerOne)),
          [1, DECK_SIZE],
        ),
        [INPUT_LVLS_B]: new ort.Tensor(
          "float32",
          Float32Array.from(levels(playerTwo)),
          [1, DECK_SIZE],
        ),
      };

      const output = await currentSession.run(feeds);
      const tensor = output[OUTPUT_LOGIT] ?? output[currentSession.outputNames[0]];
      if (!tensor) {
        throw new Error(`Model produced no "${OUTPUT_LOGIT}" output.`);
      }

      const logit = Number(tensor.data[0]);
      const probability = Math.max(0, Math.min(1, sigmoid(logit)));

      setPrediction({
        status: "complete",
        probability,
        message: "Prediction complete.",
      });
    } catch (error: unknown) {
      setPrediction({
        status: "error",
        probability: null,
        message:
          error instanceof Error
            ? error.message
            : "The model could not produce a prediction.",
      });
    }
  };

  const probabilityPercent =
    prediction.probability === null ? null : Math.round(prediction.probability * 1000) / 10;

  return (
    <main className="app-shell">
      <section className="match-header">
        <div>
          <h1>Clash Royale Win Predictor</h1>
        </div>
      </section>

      {cardLoadError ? <div className="notice error">{cardLoadError}</div> : null}

      {hiddenCardCount > 0 ? (
        <div className="notice">
          {hiddenCardCount} card{hiddenCardCount === 1 ? "" : "s"} hidden &mdash;
          {hiddenCardCount === 1 ? " it does" : " they do"} not appear in the games this
          model was trained on.
        </div>
      ) : null}

      <section className="arena-layout" aria-label="Deck matchup builder">
        <DeckBuilder
          title="Player 1's Deck"
          player="playerOne"
          deck={playerOne}
          cards={cards}
          selectedInDeck={playerOneSelected}
          activeSlot={activeSlot}
          onOpenSlot={setActiveSlot}
          onSelectCard={updateSlotCard}
          onLevelChange={updateSlotLevel}
          onClearSlot={clearSlot}
        />
        <PredictionPanel
          isComplete={isComplete}
          modelReady={manifest !== null}
          prediction={prediction}
          probabilityPercent={probabilityPercent}
          onPredict={runPrediction}
        />
        <DeckBuilder
          title="Player 2's Deck"
          player="playerTwo"
          deck={playerTwo}
          cards={cards}
          selectedInDeck={playerTwoSelected}
          activeSlot={activeSlot}
          onOpenSlot={setActiveSlot}
          onSelectCard={updateSlotCard}
          onLevelChange={updateSlotLevel}
          onClearSlot={clearSlot}
        />
      </section>
    </main>
  );
}

type DeckBuilderProps = {
  title: string;
  subtitle?: string;
  player: PlayerKey;
  deck: DeckSlot[];
  cards: Card[];
  selectedInDeck: Set<string>;
  activeSlot: { player: PlayerKey; index: number } | null;
  onOpenSlot: (slot: { player: PlayerKey; index: number } | null) => void;
  onSelectCard: (player: PlayerKey, index: number, cardName: string) => void;
  onLevelChange: (player: PlayerKey, index: number, level: number) => void;
  onClearSlot: (player: PlayerKey, index: number) => void;
};

function DeckBuilder({
  title,
  subtitle,
  player,
  deck,
  cards,
  selectedInDeck,
  activeSlot,
  onOpenSlot,
  onSelectCard,
  onLevelChange,
  onClearSlot,
}: DeckBuilderProps) {
  return (
    <section className={"deck-panel " + (player === "playerOne" ? "player-one-panel" : "player-two-panel")}>
      <header className="deck-header">
        <div>
          <h2>{title}</h2>
          {subtitle ? <p>{subtitle}</p> : null}
        </div>
        <span>{deck.filter((slot) => slot.cardName).length}/8</span>
      </header>
      <div className="deck-grid">
        {deck.map((slot, index) => {
          const selectedCard = cards.find((card) => card.name === slot.cardName);
          const isOpen = activeSlot?.player === player && activeSlot.index === index;

          return (
            <div className="slot-wrap" key={`${player}-${index}`}>
              <button
                className={`card-slot ${selectedCard ? "filled" : ""}`}
                type="button"
                onClick={() => onOpenSlot(isOpen ? null : { player, index })}
                aria-label={`${title} card ${index + 1}`}
                data-testid={`${player}-slot-${index}`}
              >
                {selectedCard ? (
                  <CardImage card={selectedCard} />
                ) : (
                  <span className="empty-plus">+</span>
                )}
              </button>
              <select
                className="level-select"
                value={slot.level}
                onChange={(event) => onLevelChange(player, index, Number(event.target.value))}
                aria-label={`${title} card ${index + 1} level`}
                data-testid={`${player}-level-${index}`}
                disabled={!slot.cardName}
              >
                {LEVELS.map((level) => (
                  <option value={level} key={level}>
                    L{level}
                  </option>
                ))}
              </select>
              {slot.cardName ? (
                <button
                  className="clear-slot"
                  type="button"
                  onClick={() => onClearSlot(player, index)}
                  data-testid={`${player}-clear-${index}`}
                >
                  Clear
                </button>
              ) : null}
              {isOpen ? (
                <CardPicker
                  cards={cards}
                  currentCardName={slot.cardName}
                  selectedInDeck={selectedInDeck}
                  onSelect={(cardName) => onSelectCard(player, index, cardName)}
                />
              ) : null}
            </div>
          );
        })}
      </div>
    </section>
  );
}

type CardPickerProps = {
  cards: Card[];
  currentCardName: string | null;
  selectedInDeck: Set<string>;
  onSelect: (cardName: string) => void;
};

function CardPicker({ cards, currentCardName, selectedInDeck, onSelect }: CardPickerProps) {
  return (
    <div className="picker" role="listbox">
      {cards.map((card) => {
        const isCurrent = currentCardName === card.name;
        const isUnavailable = selectedInDeck.has(card.name) && !isCurrent;

        return (
          <button
            type="button"
            className={`picker-option ${isCurrent ? "selected" : ""}`}
            disabled={isUnavailable}
            key={card.name}
            onClick={() => onSelect(card.name)}
            data-testid={`card-option-${card.tokenId}`}
          >
            <span className="picker-thumb">
              <CardImage card={card} compact />
            </span>
            <span>{card.displayName}</span>
          </button>
        );
      })}
    </div>
  );
}

function CardImage({ card, compact = false }: { card: Card; compact?: boolean }) {
  const [failed, setFailed] = useState(false);

  if (failed || !card.imageUrl) {
    return <span className={`card-fallback ${compact ? "compact" : ""}`}>{getInitials(card.name)}</span>;
  }

  return (
    <img
      src={card.imageUrl}
      alt={card.displayName}
      loading="lazy"
      onError={() => setFailed(true)}
    />
  );
}

type PredictionPanelProps = {
  isComplete: boolean;
  /** False until model-manifest.json has loaded; without it there is no model
   *  path to hand onnxruntime. */
  modelReady: boolean;
  prediction: PredictionState;
  probabilityPercent: number | null;
  onPredict: () => void;
};

function PredictionPanel({
  isComplete,
  modelReady,
  prediction,
  probabilityPercent,
  onPredict,
}: PredictionPanelProps) {
  const isBusy = prediction.status === "loading-model" || prediction.status === "running";
  const playerOneBarWidth = probabilityPercent ?? 0;
  const playerTwoBarWidth = probabilityPercent === null ? 0 : 100 - probabilityPercent;

  return (
    <section className="prediction-panel">
      <div className="score-label">Player 1 Win Probability</div>
      <div className="score-value" data-testid="score-value">
        {probabilityPercent === null ? "--" : `${probabilityPercent}%`}
      </div>
      <div className={`probability-track ${probabilityPercent === null ? "" : "has-prediction"}`} aria-hidden="true">
        <div
          className="probability-segment player-one-probability"
          style={{ width: `${playerOneBarWidth}%` }}
        />
        <div
          className="probability-segment player-two-probability"
          style={{ width: `${playerTwoBarWidth}%` }}
        />
      </div>
      <p className={`prediction-message ${prediction.status === "error" ? "error-text" : ""}`}>
        {prediction.message}
      </p>
      <button
        className="predict-button"
        type="button"
        disabled={!isComplete || !modelReady || isBusy}
        onClick={onPredict}
        data-testid="predict-button"
      >
        {isBusy ? "Predicting..." : "Predict"}
      </button>
    </section>
  );
}

export default App;
