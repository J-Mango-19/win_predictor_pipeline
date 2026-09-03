import { useEffect, useMemo, useState } from "react";
import * as ort from "onnxruntime-web/wasm";

const DECK_SIZE = 8;
const DEFAULT_LEVEL = 11;
const assetPath = (path: string) => `${import.meta.env.BASE_URL}${path}`;
const MODEL_PATH = assetPath("model/clash_model.onnx");
const MODEL_DATA_PATH = assetPath("model/clash_model.onnx.data");
const CARD_DATA_PATH = assetPath("data/card_to_token_id.json");
const CARD_PNG_PATH = assetPath("data/png_urls.json");
const LEVELS = Array.from({ length: 16 }, (_, index) => index + 1);

type CardMap = Record<string, number>;
type PngUrlMap = Record<string, string>;

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

const toCards = (cardMap: CardMap, pngUrls: PngUrlMap): Card[] =>
  Object.entries(cardMap)
    .map(([name, tokenId]) => {
      const displayName = cardDisplayName(name);

      return {
        name,
        tokenId,
        displayName,
        imageUrl: pngUrls[imageKey(name)] ?? "",
      };
    })
    .sort((a, b) => a.tokenId - b.tokenId);

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
  const [prediction, setPrediction] = useState<PredictionState>({
    status: "idle",
    probability: null,
    message: "Choose 16 cards to compare the decks.",
  });

  useEffect(() => {
    let cancelled = false;

    Promise.all([
      fetch(CARD_DATA_PATH).then((response) => {
        if (!response.ok) {
          throw new Error(`Card list failed to load: ${response.status}`);
        }
        return response.json() as Promise<CardMap>;
      }),
      fetch(CARD_PNG_PATH).then((response) => {
        if (!response.ok) {
          throw new Error(`Card image URLs failed to load: ${response.status}`);
        }
        return response.json() as Promise<PngUrlMap>;
      }),
    ])
      .then(([cardMap, pngUrls]) => {
        if (!cancelled) {
          setCards(toCards(cardMap, pngUrls));
        }
      })
      .catch((error: unknown) => {
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

    setPrediction({
      status: "loading-model",
      probability: null,
      message: "Loading the ONNX model in your browser...",
    });

    const nextSession = await ort.InferenceSession.create(MODEL_PATH, {
      executionProviders: ["wasm"],
      externalData: [
        {
          path: "clash_model.onnx.data",
          data: MODEL_DATA_PATH,
        },
      ],
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

      console.log("INPUT NAMES:");
      console.log(currentSession.inputNames);

      console.log("INPUT METADATA:");
      console.log(currentSession.inputMetadata);

      setPrediction({
        status: "running",
        probability: null,
        message: "Running deck matchup inference...",
      });


      const AIds = playerOne.map((slot) => {
      const card = slot.cardName ? cardByName.get(slot.cardName) : null;
      if (!card) {
        throw new Error("A selected card is missing from the token map.");
      }
      return BigInt(card.tokenId);
      });

      const BIds = playerTwo.map((slot) => {
      const card = slot.cardName ? cardByName.get(slot.cardName) : null;
      if (!card) {
        throw new Error("A selected card is missing from the token map.");
      }
      return BigInt(card.tokenId);
      });

      const ALvls = playerOne.map((slot) => slot.level / 16.0);
      const BLvls = playerTwo.map((slot) => slot.level / 16.0);

      const feeds: Record<string, ort.Tensor> = {
        A_ids: new ort.Tensor(
          "int64",
          BigInt64Array.from(AIds),
          [1, 8]
        ),

        B_ids: new ort.Tensor(
          "int64",
          BigInt64Array.from(BIds),
          [1, 8]
        ),

        A_lvls: new ort.Tensor(
          "float32",
          Float32Array.from(ALvls),
          [1, 8]
        ),

        B_lvls: new ort.Tensor(
          "float32",
          Float32Array.from(BLvls),
          [1, 8]
        ),
      };

      const output = await currentSession.run(feeds);
      const tensor = output.squeeze_1 ?? output[currentSession.outputNames[0]];

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
  prediction: PredictionState;
  probabilityPercent: number | null;
  onPredict: () => void;
};

function PredictionPanel({
  isComplete,
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
        disabled={!isComplete || isBusy}
        onClick={onPredict}
        data-testid="predict-button"
      >
        {isBusy ? "Predicting..." : "Predict"}
      </button>
    </section>
  );
}

export default App;
