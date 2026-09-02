import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()

        self.w12 = nn.Linear(d_model, 2 * d_ff, bias=False)
        self.w3  = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        gate, value = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(gate) * value)

class LevelMaskedMAB(nn.Module):
    """
    Cross-attention where query token i may only attend to key token j
    if lvl_j > lvl_i. A learned sink token is never masked out; when
    no valid card-to-card level interaction exists, all attention mass
    routes to the sink, whose V is zero-initialized (silent by default).
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads

        self.norm1_x = nn.RMSNorm(d_model)
        self.norm1_y = nn.RMSNorm(d_model)
        self.norm2   = nn.RMSNorm(d_model)

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        d_ff = d_model * 3
        self.ff      = SwiGLU(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)

        # Sink lives in post-projection K/V space (not fed through W_k / W_v).
        # sink_k: small-random init so queries can learn to "find" it.
        # sink_v: zero init so attending to the sink is a no-op at the start.
        self.sink_k = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.sink_v = nn.Parameter(torch.zeros(1, 1, d_model))

    def forward(
        self,
        X:     torch.Tensor,  # (B, n, d)  queries     (e.g. deck A cards)
        Y:     torch.Tensor,  # (B, m, d)  keys/values (e.g. deck B cards)
        lvl_x: torch.Tensor,  # (B, n)     levels of X tokens
        lvl_y: torch.Tensor,  # (B, m)     levels of Y tokens
    ) -> torch.Tensor:        # (B, n, d)

        B, n, _ = X.shape
        m = Y.shape[1]

        X_norm = self.norm1_x(X)
        Y_norm = self.norm1_y(Y)

        def split_heads(t):
            return t.view(B, -1, self.n_heads, self.d_head).transpose(1, 2)

        Q       = split_heads(self.W_q(X_norm))  # (B, h, n, d_head)
        K_cards = self.W_k(Y_norm)               # (B, m, d)
        V_cards = self.W_v(Y_norm)               # (B, m, d)

        # Prepend sink to K and V.
        # Sink params are in the same post-projection space as W_k/W_v outputs.
        K = split_heads(torch.cat([self.sink_k.expand(B, -1, -1), K_cards], dim=1))  # (B, h, m+1, d_head)
        V = split_heads(torch.cat([self.sink_v.expand(B, -1, -1), V_cards], dim=1))  # (B, h, m+1, d_head)

        # Boolean attn_mask: True = may attend, False = set to -inf before softmax.
        # Column 0 (sink): always True.
        # Columns 1..m: True iff key card j is strictly higher level than query card i.
        card_mask = lvl_y.unsqueeze(1) > lvl_x.unsqueeze(2)              # (B, n, m)
        sink_col  = torch.ones(B, n, 1, dtype=torch.bool, device=X.device)
        attn_mask = torch.cat([sink_col, card_mask], dim=2).unsqueeze(1)  # (B, 1, n, m+1)

        out = F.scaled_dot_product_attention(Q, K, V, attn_mask=attn_mask)
        out = out.transpose(1, 2).contiguous().view(B, n, -1)
        out = self.W_o(out)

        H = X + self.dropout(out)
        return H + self.dropout(self.ff(self.norm2(H)))

class MAB(nn.Module):
    """Multihead Attention Block with Pre-Norm and SwiGLU"""
    def __init__(self, d_model: int, n_heads: int, swiglu: bool=True, dropout: float=0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.use_swiglu = swiglu

        # Pre-normalization layers
        self.norm1_x = nn.RMSNorm(d_model)
        self.norm1_y = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)

        # Attention projections
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        # SwiGLU Feed-Forward (using d_model * 2 as the intermediate dimension)
        self.dropout = nn.Dropout(dropout)
        if self.use_swiglu:
            d_ff = d_model * 3
            self.ff = SwiGLU(d_model, d_ff)


    def forward(self, X: torch.Tensor, Y: torch.Tensor = None) -> torch.Tensor:
        """
        X: (B, n, d) — queries
        Y: (B, m, d) — keys and values
        if Y is None, then perform self-attention on X
        Returns: (B, n, d)
        """
        B, n, d = X.shape

        # --- 1. Pre-Norm & Attention Path ---
        X_norm = self.norm1_x(X)
        if Y is not None:
            Y_norm = self.norm1_y(Y)
        else:
            Y_norm = X_norm  # self-attention case

        def split_heads(t):
            # (B, seq, d) -> (B, h, seq, d_head)
            return t.view(B, -1, self.n_heads, self.d_head).transpose(1, 2)

        Q = split_heads(self.W_q(X_norm))   # (B, h, n, d_head)
        K = split_heads(self.W_k(Y_norm))   # (B, h, m, d_head)
        V = split_heads(self.W_v(Y_norm))   # (B, h, m, d_head)

        # Highly optimized scaled dot product attention
        out = F.scaled_dot_product_attention(Q, K, V)
        out = out.transpose(1, 2).contiguous().view(B, n, d)  # (B, n, d)
        out = self.W_o(out)

        # First residual connection (applied to raw X input)
        H = X + self.dropout(out)

        if self.use_swiglu:
            ff_out = self.ff(self.norm2(H))
            # Second residual connection
            return H + self.dropout(ff_out)
        else: 
            return H

class SXA_block(nn.Module):
    """
    Contains two sub-modules:
        * self-attention (cards within a deck attend to each other)
        * cross-attention (cards attend to cards in the other deck)
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.self_attn  = MAB(d_model, n_heads, dropout=dropout)
        self.cross_attn = MAB(d_model, n_heads, dropout=dropout)

class TransformerBinaryClassifier(nn.Module):
    """
    Input:
        1. Two sets (decks) of 8 card IDs, concatenated into one vector
        2. The levels of the cards
    """

    def __init__(
        self,
        d_model: int = 768,       # internal feature dimension
        n_heads: int = 12,        # attention heads (d_model must be divisible)
        n_sxa_blocks: int = 6,    # each block holds a self-attention & x-attention block
        n_masked_xa_blocks: int=1,# cross attn w/ attn scores corresponding to <= lvl reference cards masked out
        mlp_hidden: int = 1024,   # hidden size of the classification MLP
        vocab_size: int = 176,    # number of possible cards
        dropout: float = 0.0,     # 
        set_size: int = 8,        # number of tokens per set (ie, deck)
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_sxa_blocks = n_sxa_blocks
        self.n_masked_xa_blocks = n_masked_xa_blocks
        self.mlp_hidden = mlp_hidden
        self.set_size = set_size

        # deck summary vector
        self.deck_emb = nn.Parameter(torch.empty(1, 1, d_model))
        nn.init.normal_(self.deck_emb, std=0.02)

        # deck & inverse level (ie, "weakness") embeddings
        self.card_embedding_lookup       = nn.Embedding(vocab_size, d_model)
        self.weakness_embedding_lookup_1 = nn.Embedding(vocab_size, d_model)
        self.weakness_embedding_lookup_2 = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.card_embedding_lookup.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.weakness_embedding_lookup_1.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.weakness_embedding_lookup_2.weight, mean=0.0, std=0.01)

        # encoder: series of SXA blocks followed by attention pooling
        self.sxa_blocks = nn.ModuleList(
            [SXA_block(d_model, n_heads, dropout) for _ in range(n_sxa_blocks)]
        )

        self.masked_xa_blocks = nn.ModuleList(
            [LevelMaskedMAB(d_model, n_heads, dropout) for _ in range(n_masked_xa_blocks)]
        )

        # pooler: aggregates all card representations in a deck into a single deck embedding
        self.deck_pooler = MAB(d_model, n_heads, dropout=dropout, swiglu=False)

        self.final_norm = nn.RMSNorm(d_model)

        # classification head
        self.deck_diff_encoder = nn.Sequential(
            nn.Linear(4 * d_model, mlp_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1, bias=False)
        )


    def forward(self,
                deck_A:      torch.Tensor, # (b, set_size) card IDs for deck_A
                deck_B:      torch.Tensor, # (b, set_size) card IDs for deck_B
                deck_A_lvls: torch.Tensor, # (b, set_size) card_IDs for deck_A
                deck_B_lvls: torch.Tensor  # (b, set_size) card IDs for deck_B
                )-> torch.Tensor:
        """ Returns the probability of deck A's victory as a binary logit """

        assert deck_A.dtype in (torch.int32, torch.int64), f"incorrect dtype for deck_A: {deck_A.dtype}"
        assert deck_B.dtype in (torch.int32, torch.int64), f"incorrect dtype for deck_B: {deck_B.dtype}"
        assert deck_A_lvls.dtype == torch.float32, f"incorrect dtype for deck_A_lvls: {deck_A_lvls.dtype}"
        assert deck_B_lvls.dtype == torch.float32, f"incorrect dtype for deck_B_lvls: {deck_A_lvls.dtype}"

        b = deck_A.shape[0]


        # (b, set_size, d_model)
        deck_A_base_embs   = self.card_embedding_lookup(deck_A)
        deck_B_base_embs   = self.card_embedding_lookup(deck_B)
        deck_A_wkns_embs_1 = self.weakness_embedding_lookup_1(deck_A)
        deck_A_wkns_embs_2 = self.weakness_embedding_lookup_2(deck_A)
        deck_B_wkns_embs_1 = self.weakness_embedding_lookup_1(deck_B)
        deck_B_wkns_embs_2 = self.weakness_embedding_lookup_2(deck_B)

        # model input embeddings: (b, set_size, d_model)
        weakness_A = (1 - deck_A_lvls).unsqueeze(-1)
        weakness_B = (1 - deck_B_lvls).unsqueeze(-1)
        A = deck_A_base_embs \
                + weakness_A * deck_A_wkns_embs_1 \
                + weakness_A.pow(2) * deck_A_wkns_embs_2
        B = deck_B_base_embs \
                + weakness_B * deck_B_wkns_embs_1 \
                + weakness_B.pow(2) * deck_B_wkns_embs_2

        # decks self-attend & cross-attend to each other
        for block in self.sxa_blocks:
            A = block.self_attn(A)
            B = block.self_attn(B)

            pre_xa_A, pre_xa_B = A, B

            A = block.cross_attn(pre_xa_A, pre_xa_B)
            B = block.cross_attn(pre_xa_B, pre_xa_A)
        
        for block in self.masked_xa_blocks: 
            pre_xa_A, pre_xa_B = A, B
            A = block(pre_xa_A, pre_xa_B, deck_A_lvls, deck_B_lvls)
            B = block(pre_xa_B, pre_xa_A, deck_B_lvls, deck_A_lvls)

        # use attention pooling to condense each deck's card representations into a single vector
        # (B, d_model)
        A_repr = self.final_norm(self.deck_pooler(self.deck_emb.expand(b, -1, -1), A).squeeze(1))
        B_repr = self.final_norm(self.deck_pooler(self.deck_emb.expand(b, -1, -1), B).squeeze(1))

        # --- 
        # all ops up to this point have been translation equivariant across decks
        # ie if we swapped the inputs (deck_A <-> deck_B) and (deck_A_lvls <-> deck_B_lvls)
        # then the deck reprs would also swap (A_repr <-> B_repr)
        # ---

        # ---
        # preserve antisymmetry, ie P_win(A, B) = 1 - P_win(B, A)
        # => classifier(A, B) = -classifer(B, A) where classifier outputs a pre-logit
        # A_logit = 0.5 * (logit_win(A, B) - logit_win(B, A)) satisfies this
        # ---


        A_deck_repr = torch.cat([A_repr, B_repr, A_repr-B_repr, A_repr*B_repr], dim=1)
        B_deck_repr = torch.cat([B_repr, A_repr, B_repr-A_repr, A_repr*B_repr] , dim=1)

        A_pre_logit = self.deck_diff_encoder(A_deck_repr)
        B_pre_logit = self.deck_diff_encoder(B_deck_repr)

        AB_lvl_diff = deck_A_lvls.mean(dim=1, keepdims=True) - deck_B_lvls.mean(dim=1, keepdims=True)
        AB_lvl_diff = AB_lvl_diff ** 5

        A_logit = 0.5 * (A_pre_logit - B_pre_logit)
        return A_logit


# So next time I train, I'll have...
# a more expressive model, especially for lvl-based interactions (owing to the extra layer and the lvl diff mask)
# more data (continue collecting dataset)
# cleaned data (no high lvl decks losing to low lvl decks)
# more training steps (as the model becomes more underparameterized, I can run more epochs)
# bigger batch size (using 4x A100s and torch DDP)

# this lvl-based interaction thing removes the need for complex card-level embedding schemes 


# does this solve my problem?
# problem is, very small and insignificant changes to decks alter match outcomes drastically, and not in the right way
# The bigger dataset and batch size will be my main levers here.
# That's improvements across three axes for my training dataset
#     - More examples seen overall (should help generalization)
#     - More steps overall (should help specialization)
#     - Bigger batch sizes (reduction in label noise should help to learn intricate interactions)

# Linear probing and TSNE/PCA tell me my representations are good, so it must be more of an attention problem
