"""
Tactical State Prediction: S(t) -> S(t+1) Discrete State Transition Modeling

This module implements true tactical state prediction by mapping continuous features
to discrete tactical states and modeling state-to-state transitions.

Current System Limitation:
    The existing system predicts raw feature values directly (F(t) -> F(t+1)),
    which loses the higher-level tactical interpretation.

This Module Adds:
    1. TacticalStateEncoder: Maps continuous features to discrete tactical states
    2. StateTransitionModel: Models P(S(t+1) | S(t), context) with Markov + neural
    3. TacticalStatePredictor: Full prediction system with uncertainty estimation
    4. Training utilities for state prediction with proper losses

Tactical States:
    - stable_defensive: Low conflict intensity, minimal territorial changes
    - active_defense: Moderate conflict, holding territory
    - contested_low: Low-level contested operations
    - contested_high: High-level contested operations
    - offensive_preparation: Building for offensive
    - offensive_active: Active offensive operations
    - major_offensive: Large-scale offensive
    - transition: Transitional/unstable state

Architecture:
    Features(t) -> StateEncoder -> S(t) -> TransitionModel -> P(S(t+1))
                                      |
                                      +-> Context embedding for neural transitions

Integration:
    - Uses existing HAN domain encoders for feature extraction
    - Compatible with unified interpolation model outputs
    - Extends training_config for state prediction hyperparameters
"""

import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Union
import math
import json

import numpy as np

# Add analysis directory to path
ANALYSIS_DIR = Path(__file__).parent
sys.path.insert(0, str(ANALYSIS_DIR))

# PyTorch imports
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("PyTorch not available. Install with: pip install torch")

# Import from existing modules
try:
    from hierarchical_attention_network import (
        DOMAIN_CONFIGS,
        TOTAL_FEATURES,
        DomainEncoder,
        CrossDomainAttention,
        TemporalEncoder,
        PositionalEncoding
    )
    HAS_HAN = True
except ImportError:
    HAS_HAN = False
    print("Warning: Could not import HAN components")

try:
    from training_utils import WarmupCosineScheduler, GradientAccumulator
    from training_config import ExperimentConfig, TrainingConfig, DataConfig
    HAS_TRAINING_UTILS = True
except ImportError:
    HAS_TRAINING_UTILS = False
    print("Warning: Training utilities not available")

MODEL_DIR = ANALYSIS_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)


# =============================================================================
# STATE DEFINITIONS AND CONFIGURATION
# =============================================================================

@dataclass
class TacticalStateConfig:
    """Configuration for tactical state space.

    Defines the discrete states and their properties for tactical prediction.
    States are designed to capture meaningful operational phases that have
    different dynamics and implications.
    """
    n_states: int = 8
    state_names: List[str] = field(default_factory=lambda: [
        'stable_defensive',
        'active_defense',
        'contested_low',
        'contested_high',
        'offensive_preparation',
        'offensive_active',
        'major_offensive',
        'transition'
    ])

    # State embedding dimension
    state_embed_dim: int = 64

    # Temperature for Gumbel-Softmax (annealed during training)
    initial_temperature: float = 1.0
    min_temperature: float = 0.5
    temperature_anneal_rate: float = 0.995

    # Transition matrix regularization
    transition_smoothness: float = 0.1  # Encourage smooth transitions
    self_loop_prior: float = 0.3  # Prior probability of staying in same state


# Detailed state definitions for interpretability
TACTICAL_STATE_DEFINITIONS = {
    'stable_defensive': {
        'description': 'Low conflict intensity, minimal territorial changes',
        'indicators': ['Low fire activity', 'Stable front lines', 'Low casualty rates'],
        'typical_duration_days': '14-30',
        'likely_transitions': ['active_defense', 'contested_low']
    },
    'active_defense': {
        'description': 'Moderate conflict with defensive posture, holding territory',
        'indicators': ['Moderate fire activity', 'Minor front line adjustments', 'Moderate casualties'],
        'typical_duration_days': '7-21',
        'likely_transitions': ['stable_defensive', 'contested_low', 'contested_high']
    },
    'contested_low': {
        'description': 'Low-level contested operations across fronts',
        'indicators': ['Elevated fire activity', 'Small territorial exchanges', 'Increased casualties'],
        'typical_duration_days': '7-14',
        'likely_transitions': ['active_defense', 'contested_high', 'offensive_preparation']
    },
    'contested_high': {
        'description': 'High-intensity contested operations',
        'indicators': ['High fire activity', 'Active territorial changes', 'High casualty rates'],
        'typical_duration_days': '7-14',
        'likely_transitions': ['contested_low', 'offensive_active', 'transition']
    },
    'offensive_preparation': {
        'description': 'Building up for offensive operations',
        'indicators': ['Increased logistics', 'Force concentration', 'Intelligence activity'],
        'typical_duration_days': '7-21',
        'likely_transitions': ['offensive_active', 'contested_high', 'active_defense']
    },
    'offensive_active': {
        'description': 'Active offensive operations underway',
        'indicators': ['Very high fire activity', 'Territorial gains', 'High equipment losses'],
        'typical_duration_days': '3-14',
        'likely_transitions': ['contested_high', 'major_offensive', 'transition']
    },
    'major_offensive': {
        'description': 'Large-scale offensive with significant territorial changes',
        'indicators': ['Extreme fire activity', 'Rapid territorial changes', 'Very high casualties'],
        'typical_duration_days': '3-10',
        'likely_transitions': ['offensive_active', 'contested_high', 'transition']
    },
    'transition': {
        'description': 'Transitional or unstable state between operational phases',
        'indicators': ['Mixed signals', 'Rapidly changing conditions', 'Uncertain patterns'],
        'typical_duration_days': '1-7',
        'likely_transitions': ['any']
    }
}


@dataclass
class TacticalPredictionConfig:
    """Configuration for the full tactical prediction system."""
    # State space config
    state_config: TacticalStateConfig = field(default_factory=TacticalStateConfig)

    # Model architecture
    hidden_dim: int = 128
    context_dim: int = 64
    nhead: int = 4
    num_layers: int = 2
    dropout: float = 0.1

    # Training
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_epochs: int = 10
    max_epochs: int = 200
    patience: int = 30
    batch_size: int = 16

    # Loss weights
    state_loss_weight: float = 1.0
    transition_loss_weight: float = 1.0
    consistency_loss_weight: float = 0.1
    uncertainty_loss_weight: float = 0.1


# =============================================================================
# NEURAL NETWORK COMPONENTS
# =============================================================================

if HAS_TORCH:

    class StatePrototypes(nn.Module):
        """
        Learnable state prototypes for soft state assignment.

        Each tactical state is represented by a learnable prototype vector.
        State assignment is computed via similarity to prototypes.
        """

        def __init__(self, n_states: int, embed_dim: int):
            super().__init__()
            self.n_states = n_states
            self.embed_dim = embed_dim

            # Learnable prototype vectors for each state
            self.prototypes = nn.Parameter(torch.randn(n_states, embed_dim))

            # Initialize prototypes to be spread out
            nn.init.orthogonal_(self.prototypes)

        def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
            """
            Compute similarity between embeddings and prototypes.

            Args:
                embeddings: [batch, embed_dim] - encoded feature embeddings

            Returns:
                [batch, n_states] - similarity scores (logits)
            """
            # Normalize for cosine similarity
            emb_norm = F.normalize(embeddings, p=2, dim=-1)
            proto_norm = F.normalize(self.prototypes, p=2, dim=-1)

            # Cosine similarity as logits
            similarities = torch.matmul(emb_norm, proto_norm.T)

            return similarities

        def get_prototype(self, state_idx: int) -> torch.Tensor:
            """Get the prototype vector for a given state."""
            return self.prototypes[state_idx]


    class TacticalStateEncoder(nn.Module):
        """
        Encodes multi-source features into discrete tactical states.

        States represent high-level operational phases:
        - Conflict intensity levels (low, medium, high, major_offensive)
        - Territorial control regimes (stable, contested, shifting)
        - Operational phases (defensive, offensive, stalemate)

        Uses soft state assignment during training (Gumbel-Softmax) and
        hard assignment during inference.
        """

        def __init__(
            self,
            input_dim: int,
            n_states: int = 8,
            hidden_dim: int = 128,
            state_embed_dim: int = 64,
            dropout: float = 0.1,
            use_prototypes: bool = True
        ):
            super().__init__()
            self.input_dim = input_dim
            self.n_states = n_states
            self.hidden_dim = hidden_dim
            self.state_embed_dim = state_embed_dim
            self.use_prototypes = use_prototypes

            # Feature encoder: projects raw features to hidden space
            self.feature_encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )

            # State classification head
            if use_prototypes:
                # Project to embedding space for prototype comparison
                self.state_projection = nn.Linear(hidden_dim, state_embed_dim)
                self.prototypes = StatePrototypes(n_states, state_embed_dim)
            else:
                # Direct classification
                self.state_classifier = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim // 2, n_states)
                )

            # Learnable state embeddings (for downstream use)
            self.state_embeddings = nn.Embedding(n_states, state_embed_dim)

            # Temperature parameter for Gumbel-Softmax
            self.register_buffer('temperature', torch.tensor(1.0))

        def encode_features(self, features: torch.Tensor) -> torch.Tensor:
            """
            Encode raw features into hidden representation.

            Args:
                features: [batch, input_dim] or [batch, seq_len, input_dim]

            Returns:
                [batch, hidden_dim] or [batch, seq_len, hidden_dim]
            """
            return self.feature_encoder(features)

        def get_state_logits(self, encoded: torch.Tensor) -> torch.Tensor:
            """
            Get state classification logits from encoded features.

            Args:
                encoded: [batch, hidden_dim] - encoded features

            Returns:
                [batch, n_states] - state logits
            """
            if self.use_prototypes:
                state_emb = self.state_projection(encoded)
                logits = self.prototypes(state_emb)
            else:
                logits = self.state_classifier(encoded)

            return logits

        def forward(
            self,
            features: torch.Tensor,
            hard: bool = False,
            return_embedding: bool = True
        ) -> Dict[str, torch.Tensor]:
            """
            Encode features and classify into tactical states.

            Args:
                features: [batch, input_dim] or [batch, seq_len, input_dim]
                hard: If True, use hard (argmax) assignment; else use soft (Gumbel-Softmax)
                return_embedding: If True, return state embedding

            Returns:
                Dictionary with:
                - 'logits': [batch, n_states] - raw logits
                - 'probs': [batch, n_states] - softmax probabilities
                - 'state_dist': [batch, n_states] - Gumbel-Softmax (soft) or one-hot (hard)
                - 'state_embedding': [batch, state_embed_dim] - weighted state embedding
                - 'encoded': [batch, hidden_dim] - encoded features
            """
            # Handle sequence input
            input_shape = features.shape
            if len(input_shape) == 3:
                batch_size, seq_len, _ = input_shape
                features_flat = features.view(-1, self.input_dim)
            else:
                features_flat = features
                batch_size = features.size(0)
                seq_len = None

            # Encode features
            encoded = self.encode_features(features_flat)

            # Get state logits
            logits = self.get_state_logits(encoded)

            # Compute probabilities
            probs = F.softmax(logits, dim=-1)

            # Gumbel-Softmax for differentiable discrete sampling
            if hard or not self.training:
                # Hard assignment (argmax)
                state_idx = logits.argmax(dim=-1)
                state_dist = F.one_hot(state_idx, self.n_states).float()
            else:
                # Soft assignment (Gumbel-Softmax)
                state_dist = F.gumbel_softmax(logits, tau=self.temperature.item(), hard=False)

            outputs = {
                'logits': logits,
                'probs': probs,
                'state_dist': state_dist,
                'encoded': encoded
            }

            # Compute weighted state embedding
            if return_embedding:
                # Use state distribution to weight state embeddings
                state_embedding = torch.matmul(state_dist, self.state_embeddings.weight)
                outputs['state_embedding'] = state_embedding

            # Reshape outputs if sequence input
            if seq_len is not None:
                for key in outputs:
                    if outputs[key] is not None:
                        outputs[key] = outputs[key].view(batch_size, seq_len, -1)

            return outputs

        def get_state_embedding(self, state_idx: Union[int, torch.Tensor]) -> torch.Tensor:
            """
            Get the embedding for a given state index.

            Args:
                state_idx: State index (int) or indices [batch]

            Returns:
                [state_embed_dim] or [batch, state_embed_dim]
            """
            if isinstance(state_idx, int):
                state_idx = torch.tensor([state_idx], device=self.state_embeddings.weight.device)
            return self.state_embeddings(state_idx)

        def anneal_temperature(self, rate: float = 0.995, min_temp: float = 0.5):
            """Anneal the Gumbel-Softmax temperature."""
            new_temp = max(self.temperature.item() * rate, min_temp)
            self.temperature.fill_(new_temp)


    class StateTransitionModel(nn.Module):
        """
        Models transition probabilities between tactical states.

        Combines:
        1. Learned base transition matrix (Markov prior)
        2. Context-dependent neural transitions

        P(S(t+1) | S(t), context) = alpha * Markov(S(t)) + (1-alpha) * Neural(S(t), context)

        where alpha is learned based on context quality.
        """

        def __init__(
            self,
            n_states: int,
            context_dim: int = 64,
            hidden_dim: int = 128,
            dropout: float = 0.1,
            self_loop_prior: float = 0.3
        ):
            super().__init__()
            self.n_states = n_states
            self.context_dim = context_dim
            self.hidden_dim = hidden_dim

            # Base transition matrix (learnable Markov)
            # Initialize with self-loop prior and uniform otherwise
            init_matrix = torch.zeros(n_states, n_states)
            init_matrix.fill_((1 - self_loop_prior) / (n_states - 1))
            init_matrix.fill_diagonal_(self_loop_prior)
            # Use log for parameterization (will be softmaxed)
            self.transition_logits = nn.Parameter(torch.log(init_matrix + 1e-8))

            # Context encoder (processes state embedding + external context)
            self.context_encoder = nn.Sequential(
                nn.Linear(context_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU()
            )

            # Neural transition predictor
            self.transition_predictor = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, n_states)
            )

            # Mixing weight predictor (learns when to use Markov vs neural)
            self.mixing_weight = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid()
            )

            # State embedding projection (if context_dim != state embedding dim)
            self.state_projection = nn.Linear(context_dim, context_dim)

        def get_transition_matrix(self) -> torch.Tensor:
            """
            Get the learned base transition matrix.

            Returns:
                [n_states, n_states] - row-normalized transition probabilities
            """
            return F.softmax(self.transition_logits, dim=-1)

        def markov_transition(
            self,
            current_state_dist: torch.Tensor
        ) -> torch.Tensor:
            """
            Compute next state distribution using Markov transition.

            Args:
                current_state_dist: [batch, n_states] - current state distribution

            Returns:
                [batch, n_states] - next state distribution
            """
            # P(S') = sum_S P(S' | S) * P(S)
            trans_matrix = self.get_transition_matrix()
            return torch.matmul(current_state_dist, trans_matrix)

        def neural_transition(
            self,
            current_state_embedding: torch.Tensor,
            context: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Compute context-dependent transition distribution.

            Args:
                current_state_embedding: [batch, context_dim] - current state embedding
                context: [batch, context_dim] - external context (e.g., from HAN)

            Returns:
                next_state_logits: [batch, n_states]
                mixing_weight: [batch, 1] - how much to trust neural vs Markov
            """
            # Project state embedding
            state_proj = self.state_projection(current_state_embedding)

            # Encode combined context
            combined = torch.cat([state_proj, context], dim=-1)
            encoded = self.context_encoder(combined)

            # Predict transition distribution
            next_logits = self.transition_predictor(encoded)

            # Predict mixing weight
            alpha = self.mixing_weight(encoded)

            return next_logits, alpha

        def forward(
            self,
            current_state_dist: torch.Tensor,
            current_state_embedding: torch.Tensor,
            context: Optional[torch.Tensor] = None
        ) -> Dict[str, torch.Tensor]:
            """
            Predict next state distribution.

            Args:
                current_state_dist: [batch, n_states] - current state probabilities
                current_state_embedding: [batch, context_dim] - current state embedding
                context: [batch, context_dim] - optional external context

            Returns:
                Dictionary with:
                - 'next_state_dist': [batch, n_states] - predicted next state distribution
                - 'markov_dist': [batch, n_states] - Markov-only prediction
                - 'neural_dist': [batch, n_states] - Neural-only prediction
                - 'mixing_weight': [batch, 1] - learned mixing weight
            """
            # Markov transition
            markov_dist = self.markov_transition(current_state_dist)

            if context is None:
                # No context - use pure Markov
                return {
                    'next_state_dist': markov_dist,
                    'markov_dist': markov_dist,
                    'neural_dist': markov_dist,
                    'mixing_weight': torch.zeros(current_state_dist.size(0), 1,
                                                   device=current_state_dist.device)
                }

            # Neural transition with context
            neural_logits, alpha = self.neural_transition(current_state_embedding, context)
            neural_dist = F.softmax(neural_logits, dim=-1)

            # Mix Markov and neural predictions
            # alpha controls how much to trust context (high alpha = more neural)
            next_state_dist = (1 - alpha) * markov_dist + alpha * neural_dist

            return {
                'next_state_dist': next_state_dist,
                'markov_dist': markov_dist,
                'neural_dist': neural_dist,
                'mixing_weight': alpha,
                'neural_logits': neural_logits
            }


    class UncertaintyHead(nn.Module):
        """
        Estimates epistemic uncertainty in state predictions.

        Uses a separate network to predict uncertainty, which helps
        identify when predictions may be unreliable.
        """

        def __init__(self, input_dim: int, hidden_dim: int = 64, n_states: int = 8):
            super().__init__()
            self.n_states = n_states

            self.network = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, n_states),
                nn.Softplus()  # Positive uncertainty
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Predict uncertainty for each state.

            Args:
                x: [batch, input_dim] - encoded features

            Returns:
                [batch, n_states] - uncertainty (standard deviation) per state
            """
            return self.network(x) + 0.01  # Minimum uncertainty floor


    class TacticalStatePredictor(nn.Module):
        """
        Complete tactical state prediction system.

        Flow:
            Features(t) -> StateEncoder -> S(t) -> TransitionModel -> P(S(t+1))

        Supports:
        - Single-step prediction: S(t) -> S(t+1)
        - Multi-step trajectory: S(t) -> S(t+1) -> ... -> S(t+H)
        - Uncertainty estimation for predictions
        """

        def __init__(
            self,
            feature_dim: int,
            config: TacticalPredictionConfig
        ):
            super().__init__()
            self.feature_dim = feature_dim
            self.config = config
            state_config = config.state_config

            # State encoder
            self.state_encoder = TacticalStateEncoder(
                input_dim=feature_dim,
                n_states=state_config.n_states,
                hidden_dim=config.hidden_dim,
                state_embed_dim=state_config.state_embed_dim,
                dropout=config.dropout,
                use_prototypes=True
            )

            # Transition model
            self.transition_model = StateTransitionModel(
                n_states=state_config.n_states,
                context_dim=state_config.state_embed_dim,
                hidden_dim=config.hidden_dim,
                dropout=config.dropout,
                self_loop_prior=state_config.self_loop_prior
            )

            # Uncertainty estimation
            self.uncertainty_head = UncertaintyHead(
                input_dim=config.hidden_dim,
                hidden_dim=config.hidden_dim // 2,
                n_states=state_config.n_states
            )

            # Context projection (from encoded features to context for transition)
            self.context_projection = nn.Sequential(
                nn.Linear(config.hidden_dim, state_config.state_embed_dim),
                nn.LayerNorm(state_config.state_embed_dim),
                nn.GELU()
            )

            # Temporal context encoder for sequence inputs
            self.temporal_encoder = nn.LSTM(
                input_size=config.hidden_dim,
                hidden_size=config.hidden_dim,
                num_layers=config.num_layers,
                batch_first=True,
                dropout=config.dropout if config.num_layers > 1 else 0,
                bidirectional=False
            )

        def encode_current_state(
            self,
            features: torch.Tensor,
            hard: bool = False
        ) -> Dict[str, torch.Tensor]:
            """
            Encode current features into tactical state.

            Args:
                features: [batch, feature_dim] or [batch, seq_len, feature_dim]
                hard: Use hard state assignment

            Returns:
                State encoding results from state_encoder
            """
            return self.state_encoder(features, hard=hard, return_embedding=True)

        def forward(
            self,
            features: torch.Tensor,
            return_probs: bool = True,
            hard: bool = False
        ) -> Dict[str, torch.Tensor]:
            """
            Predict next state distribution from features.

            Args:
                features: [batch, feature_dim] - current features
                return_probs: Return probability distributions
                hard: Use hard state assignment

            Returns:
                Dictionary with:
                - 'current_state': Current state distribution [batch, n_states]
                - 'next_state': Predicted next state distribution [batch, n_states]
                - 'uncertainty': Prediction uncertainty [batch, n_states]
                - 'current_embedding': Current state embedding [batch, state_embed_dim]
                - 'transition_info': Details from transition model
            """
            # Encode current state
            state_outputs = self.state_encoder(features, hard=hard, return_embedding=True)

            current_state_dist = state_outputs['probs']  # [batch, n_states]
            current_embedding = state_outputs['state_embedding']  # [batch, state_embed_dim]
            encoded = state_outputs['encoded']  # [batch, hidden_dim]

            # Generate context for transition model
            context = self.context_projection(encoded)

            # Predict next state
            transition_outputs = self.transition_model(
                current_state_dist=current_state_dist,
                current_state_embedding=current_embedding,
                context=context
            )

            # Estimate uncertainty
            uncertainty = self.uncertainty_head(encoded)

            outputs = {
                'current_state': current_state_dist,
                'current_logits': state_outputs['logits'],
                'next_state': transition_outputs['next_state_dist'],
                'uncertainty': uncertainty,
                'current_embedding': current_embedding,
                'transition_info': transition_outputs,
                'encoded': encoded
            }

            return outputs

        def predict_sequence(
            self,
            features: torch.Tensor,
            hard: bool = False
        ) -> Dict[str, torch.Tensor]:
            """
            Process a sequence of features and predict next state.

            Args:
                features: [batch, seq_len, feature_dim] - sequence of features
                hard: Use hard state assignment

            Returns:
                Predictions based on the full sequence context
            """
            batch_size, seq_len, _ = features.shape

            # Encode each timestep
            state_outputs = self.state_encoder(features, hard=hard, return_embedding=True)

            # Get encoded features [batch, seq_len, hidden_dim]
            encoded_seq = state_outputs['encoded']

            # Process with temporal encoder
            temporal_out, (h_n, c_n) = self.temporal_encoder(encoded_seq)

            # Use final hidden state as context
            final_context = self.context_projection(h_n[-1])  # [batch, state_embed_dim]

            # Get final state distribution
            final_state_dist = state_outputs['probs'][:, -1, :]  # [batch, n_states]
            final_embedding = state_outputs['state_embedding'][:, -1, :]

            # Predict next state using temporal context
            transition_outputs = self.transition_model(
                current_state_dist=final_state_dist,
                current_state_embedding=final_embedding,
                context=final_context
            )

            # Estimate uncertainty using final encoding
            uncertainty = self.uncertainty_head(temporal_out[:, -1, :])

            return {
                'state_sequence': state_outputs['probs'],  # [batch, seq_len, n_states]
                'current_state': final_state_dist,
                'next_state': transition_outputs['next_state_dist'],
                'uncertainty': uncertainty,
                'transition_info': transition_outputs,
                'temporal_context': final_context
            }

        def predict_trajectory(
            self,
            features: torch.Tensor,
            horizon: int = 7,
            return_all_distributions: bool = True
        ) -> Dict[str, torch.Tensor]:
            """
            Multi-step prediction: S(t) -> S(t+1) -> ... -> S(t+H)

            Uses the learned transition model to propagate state distributions
            forward in time.

            Args:
                features: [batch, feature_dim] or [batch, seq_len, feature_dim]
                horizon: Number of steps to predict ahead
                return_all_distributions: Return distributions at each step

            Returns:
                Dictionary with:
                - 'trajectory': [batch, horizon, n_states] - state distribution at each step
                - 'uncertainties': [batch, horizon, n_states] - uncertainty at each step
                - 'final_state': [batch, n_states] - distribution at horizon
            """
            # Handle sequence input
            if features.dim() == 3:
                outputs = self.predict_sequence(features)
                current_dist = outputs['next_state']
                current_embedding = outputs['temporal_context']
            else:
                outputs = self.forward(features)
                current_dist = outputs['next_state']
                current_embedding = outputs['current_embedding']

            trajectory = [current_dist]
            uncertainties = [outputs['uncertainty']]

            # Propagate forward
            for step in range(horizon - 1):
                # Compute expected embedding for current distribution
                weighted_embedding = torch.matmul(
                    current_dist,
                    self.state_encoder.state_embeddings.weight
                )

                # Predict next step (no context in multi-step, use Markov-weighted)
                trans_outputs = self.transition_model(
                    current_state_dist=current_dist,
                    current_state_embedding=weighted_embedding,
                    context=None  # No fresh features for future steps
                )

                current_dist = trans_outputs['next_state_dist']
                trajectory.append(current_dist)

                # Uncertainty grows with horizon
                base_uncertainty = outputs['uncertainty']
                step_uncertainty = base_uncertainty * (1 + 0.1 * (step + 1))
                uncertainties.append(step_uncertainty)

            trajectory = torch.stack(trajectory, dim=1)  # [batch, horizon, n_states]
            uncertainties = torch.stack(uncertainties, dim=1)

            return {
                'trajectory': trajectory,
                'uncertainties': uncertainties,
                'final_state': trajectory[:, -1, :],
                'initial_state': outputs.get('current_state', trajectory[:, 0, :])
            }

        def get_most_likely_trajectory(
            self,
            features: torch.Tensor,
            horizon: int = 7
        ) -> Dict[str, Any]:
            """
            Get the most likely state sequence (for interpretability).

            Args:
                features: Input features
                horizon: Prediction horizon

            Returns:
                Dictionary with state indices and names at each step
            """
            traj_outputs = self.predict_trajectory(features, horizon)
            trajectory = traj_outputs['trajectory']  # [batch, horizon, n_states]

            # Get most likely state at each step
            most_likely_indices = trajectory.argmax(dim=-1)  # [batch, horizon]
            most_likely_probs = trajectory.max(dim=-1).values  # [batch, horizon]

            state_names = self.config.state_config.state_names

            return {
                'state_indices': most_likely_indices,
                'state_names': [[state_names[idx] for idx in seq] for seq in most_likely_indices],
                'state_probabilities': most_likely_probs,
                'full_distributions': trajectory
            }


    # =========================================================================
    # TRAINING UTILITIES
    # =========================================================================

    class TacticalStateLoss(nn.Module):
        """
        Combined loss function for tactical state prediction.

        Components:
        1. State classification loss (cross-entropy or soft cross-entropy)
        2. Transition distribution loss (KL divergence)
        3. Temporal consistency loss (encourages smooth state transitions)
        4. Uncertainty calibration loss (optional)
        """

        def __init__(
            self,
            n_states: int = 8,
            state_weight: float = 1.0,
            transition_weight: float = 1.0,
            consistency_weight: float = 0.1,
            uncertainty_weight: float = 0.1,
            label_smoothing: float = 0.1
        ):
            super().__init__()
            self.n_states = n_states
            self.state_weight = state_weight
            self.transition_weight = transition_weight
            self.consistency_weight = consistency_weight
            self.uncertainty_weight = uncertainty_weight
            self.label_smoothing = label_smoothing

        def state_classification_loss(
            self,
            logits: torch.Tensor,
            targets: torch.Tensor
        ) -> torch.Tensor:
            """
            Cross-entropy loss for state classification.

            Args:
                logits: [batch, n_states] - predicted logits
                targets: [batch] - target state indices

            Returns:
                Scalar loss
            """
            return F.cross_entropy(
                logits, targets,
                label_smoothing=self.label_smoothing
            )

        def transition_kl_loss(
            self,
            predicted_dist: torch.Tensor,
            target_dist: torch.Tensor
        ) -> torch.Tensor:
            """
            KL divergence loss for transition distributions.

            Args:
                predicted_dist: [batch, n_states] - predicted next state distribution
                target_dist: [batch, n_states] - target distribution (one-hot or soft)

            Returns:
                Scalar loss
            """
            # Add small epsilon for numerical stability
            predicted_log = torch.log(predicted_dist + 1e-8)

            # KL(target || predicted)
            kl = F.kl_div(predicted_log, target_dist, reduction='batchmean')

            return kl

        def consistency_loss(
            self,
            state_sequence: torch.Tensor
        ) -> torch.Tensor:
            """
            Temporal consistency loss - penalizes rapid state changes.

            Args:
                state_sequence: [batch, seq_len, n_states] - state distributions over time

            Returns:
                Scalar loss encouraging smooth transitions
            """
            if state_sequence.size(1) < 2:
                return torch.tensor(0.0, device=state_sequence.device)

            # Compute difference between consecutive states
            diffs = state_sequence[:, 1:, :] - state_sequence[:, :-1, :]

            # L2 norm of differences (penalize large changes)
            consistency = (diffs ** 2).sum(dim=-1).mean()

            return consistency

        def uncertainty_calibration_loss(
            self,
            predictions: torch.Tensor,
            targets: torch.Tensor,
            uncertainties: torch.Tensor
        ) -> torch.Tensor:
            """
            Calibration loss - uncertainty should correlate with error.

            Args:
                predictions: [batch, n_states] - predicted distributions
                targets: [batch] - target state indices
                uncertainties: [batch, n_states] - predicted uncertainties

            Returns:
                Loss encouraging well-calibrated uncertainty
            """
            # Compute prediction error (probability assigned to true class)
            target_one_hot = F.one_hot(targets, self.n_states).float()
            errors = (predictions - target_one_hot).abs()

            # Mean uncertainty for predicted class
            mean_uncertainty = (uncertainties * predictions).sum(dim=-1)

            # Mean error
            mean_error = (errors * target_one_hot).sum(dim=-1)

            # Uncertainty should match error (squared difference)
            calibration = ((mean_uncertainty - mean_error) ** 2).mean()

            return calibration

        def forward(
            self,
            outputs: Dict[str, torch.Tensor],
            targets: Dict[str, torch.Tensor]
        ) -> Dict[str, torch.Tensor]:
            """
            Compute combined loss.

            Args:
                outputs: Dictionary from TacticalStatePredictor
                targets: Dictionary with 'current_state', 'next_state' indices

            Returns:
                Dictionary with individual losses and total
            """
            losses = {}

            # State classification loss (current state)
            if 'current_logits' in outputs and 'current_state' in targets:
                losses['state_loss'] = self.state_weight * self.state_classification_loss(
                    outputs['current_logits'],
                    targets['current_state']
                )
            else:
                losses['state_loss'] = torch.tensor(0.0, device=next(iter(outputs.values())).device)

            # Transition loss (next state prediction)
            if 'next_state' in outputs and 'next_state' in targets:
                # Convert target index to one-hot
                if targets['next_state'].dim() == 1:
                    target_dist = F.one_hot(targets['next_state'], self.n_states).float()
                else:
                    target_dist = targets['next_state']

                losses['transition_loss'] = self.transition_weight * self.transition_kl_loss(
                    outputs['next_state'],
                    target_dist
                )
            else:
                losses['transition_loss'] = torch.tensor(0.0, device=next(iter(outputs.values())).device)

            # Consistency loss (if sequence)
            if 'state_sequence' in outputs:
                losses['consistency_loss'] = self.consistency_weight * self.consistency_loss(
                    outputs['state_sequence']
                )
            else:
                losses['consistency_loss'] = torch.tensor(0.0, device=next(iter(outputs.values())).device)

            # Uncertainty calibration (if provided)
            if 'uncertainty' in outputs and 'next_state' in targets:
                losses['uncertainty_loss'] = self.uncertainty_weight * self.uncertainty_calibration_loss(
                    outputs['next_state'],
                    targets['next_state'],
                    outputs['uncertainty']
                )
            else:
                losses['uncertainty_loss'] = torch.tensor(0.0, device=next(iter(outputs.values())).device)

            # Total loss
            losses['total'] = sum(losses.values())

            return losses


    class TacticalStateDataset(Dataset):
        """
        Dataset for tactical state prediction training.

        Creates (features, current_state, next_state) pairs from time series data.
        States are derived from feature patterns using rule-based labeling
        or can be provided as ground truth labels.
        """

        def __init__(
            self,
            features: np.ndarray,
            state_labels: Optional[np.ndarray] = None,
            seq_len: int = 4,
            train: bool = True,
            val_ratio: float = 0.2,
            temporal_gap: int = 7,
            n_states: int = 8,
            shared_percentiles: Optional[np.ndarray] = None,
            shared_norm_stats: Optional[Dict] = None
        ):
            """
            Initialize dataset.

            Args:
                features: [n_timesteps, feature_dim] - time series features
                state_labels: [n_timesteps] - optional ground truth state labels
                seq_len: Number of timesteps for context
                train: Whether this is training set
                val_ratio: Validation ratio for split
                temporal_gap: Gap between train/val
                n_states: Number of states for derived labels
                shared_percentiles: Percentile boundaries from training split (required for val)
                shared_norm_stats: Normalization stats from training split (required for val)
            """
            self.seq_len = seq_len
            self.n_states = n_states

            # Compute split boundaries first
            n_samples = len(features) - seq_len
            n_train = int(n_samples * (1 - val_ratio))

            # CRITICAL FIX: Normalize features using TRAINING stats only
            if train:
                # Compute normalization on training portion only
                train_features = features[:n_train + seq_len]
                self.feature_mean = np.mean(train_features, axis=0, keepdims=True)
                self.feature_std = np.maximum(np.std(train_features, axis=0, keepdims=True), 0.1)
                self.features = (features - self.feature_mean) / self.feature_std
            else:
                # Val/Test: Use provided stats from training
                if shared_norm_stats is None:
                    raise ValueError(
                        "shared_norm_stats required for validation split. "
                        "Pass {'mean': train_ds.feature_mean, 'std': train_ds.feature_std}"
                    )
                self.feature_mean = shared_norm_stats['mean']
                self.feature_std = shared_norm_stats['std']
                self.features = (features - self.feature_mean) / self.feature_std

            # CRITICAL FIX: Derive labels using TRAINING percentiles only
            if state_labels is None:
                if train:
                    # Compute percentiles on training data only
                    train_norm_features = self.features[:n_train + seq_len]
                    intensity = np.mean(np.abs(train_norm_features), axis=1)
                    self.percentile_boundaries = np.percentile(intensity, np.linspace(0, 100, n_states + 1))
                    state_labels = self._derive_state_labels(self.features, n_states, self.percentile_boundaries)
                else:
                    # Val/Test: Use provided percentile boundaries
                    if shared_percentiles is None:
                        raise ValueError(
                            "shared_percentiles required for validation split. "
                            "Pass train_dataset.percentile_boundaries"
                        )
                    self.percentile_boundaries = shared_percentiles
                    state_labels = self._derive_state_labels(self.features, n_states, self.percentile_boundaries)
            else:
                self.percentile_boundaries = None

            self.state_labels = state_labels

            # Create samples (current window -> next state)
            n_samples = len(features) - seq_len

            # Temporal split
            n_train = int(n_samples * (1 - val_ratio))

            if train:
                self.start_idx = 0
                self.end_idx = n_train - temporal_gap
            else:
                self.start_idx = n_train
                self.end_idx = n_samples

            self.indices = list(range(self.start_idx, self.end_idx))

        def _derive_state_labels(
            self,
            features: np.ndarray,
            n_states: int,
            percentiles: np.ndarray
        ) -> np.ndarray:
            """
            Derive state labels from features using intensity-based discretization.

            Uses pre-computed percentile boundaries to ensure consistent labeling
            across train/val/test splits.

            Args:
                features: [n_timesteps, feature_dim] - normalized features
                n_states: Number of discrete states
                percentiles: Pre-computed percentile boundaries from training data

            Returns:
                labels: [n_timesteps] - integer state labels
            """
            # Compute aggregate intensity metric
            intensity = np.mean(np.abs(features), axis=1)

            # Discretize using provided percentile boundaries
            # percentiles[1:-1] gives the internal bin edges
            labels = np.digitize(intensity, percentiles[1:-1])

            return labels.astype(np.int64)

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, idx):
            actual_idx = self.indices[idx]

            # Feature window
            feature_window = self.features[actual_idx:actual_idx + self.seq_len]

            # Current state (at end of window)
            current_state = self.state_labels[actual_idx + self.seq_len - 1]

            # Next state (target)
            next_state = self.state_labels[actual_idx + self.seq_len]

            return {
                'features': torch.tensor(feature_window, dtype=torch.float32),
                'current_state': torch.tensor(current_state, dtype=torch.long),
                'next_state': torch.tensor(next_state, dtype=torch.long)
            }


    def train_tactical_predictor(
        model: TacticalStatePredictor,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TacticalPredictionConfig,
        device: str = 'cpu'
    ) -> Dict[str, List[float]]:
        """
        Training loop for tactical state predictor.

        Args:
            model: TacticalStatePredictor to train
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration
            device: Device to train on

        Returns:
            Training history with losses and metrics
        """
        model = model.to(device)

        # Loss function
        loss_fn = TacticalStateLoss(
            n_states=config.state_config.n_states,
            state_weight=config.state_loss_weight,
            transition_weight=config.transition_loss_weight,
            consistency_weight=config.consistency_loss_weight,
            uncertainty_weight=config.uncertainty_loss_weight
        )

        # Optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        # Scheduler
        if HAS_TRAINING_UTILS:
            scheduler = WarmupCosineScheduler(
                optimizer,
                warmup_epochs=config.warmup_epochs,
                total_epochs=config.max_epochs,
                warmup_start_lr=config.learning_rate * 0.01,
                min_lr=1e-7
            )
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=10
            )

        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': [],
            'transition_accuracy': []
        }

        best_val_loss = float('inf')
        patience_counter = 0

        print(f"\nTraining Tactical State Predictor for {config.max_epochs} epochs...")
        print("-" * 70)

        for epoch in range(config.max_epochs):
            # Training
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0

            for batch in train_loader:
                features = batch['features'].to(device)
                targets = {
                    'current_state': batch['current_state'].to(device),
                    'next_state': batch['next_state'].to(device)
                }

                optimizer.zero_grad()

                # Forward pass (sequence input)
                outputs = model.predict_sequence(features)

                # Compute loss
                losses = loss_fn(outputs, targets)

                losses['total'].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                train_loss += losses['total'].item()

                # Accuracy
                pred_states = outputs['current_state'].argmax(dim=-1)
                train_correct += (pred_states == targets['current_state']).sum().item()
                train_total += targets['current_state'].size(0)

            # Anneal temperature
            model.state_encoder.anneal_temperature(
                rate=config.state_config.temperature_anneal_rate,
                min_temp=config.state_config.min_temperature
            )

            # Validation
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            transition_correct = 0

            with torch.no_grad():
                for batch in val_loader:
                    features = batch['features'].to(device)
                    targets = {
                        'current_state': batch['current_state'].to(device),
                        'next_state': batch['next_state'].to(device)
                    }

                    outputs = model.predict_sequence(features)
                    losses = loss_fn(outputs, targets)

                    val_loss += losses['total'].item()

                    # Current state accuracy
                    pred_states = outputs['current_state'].argmax(dim=-1)
                    val_correct += (pred_states == targets['current_state']).sum().item()
                    val_total += targets['current_state'].size(0)

                    # Transition accuracy
                    pred_next = outputs['next_state'].argmax(dim=-1)
                    transition_correct += (pred_next == targets['next_state']).sum().item()

            # Record history
            train_loss_avg = train_loss / len(train_loader)
            val_loss_avg = val_loss / len(val_loader)
            train_acc = train_correct / train_total
            val_acc = val_correct / val_total
            trans_acc = transition_correct / val_total

            history['train_loss'].append(train_loss_avg)
            history['val_loss'].append(val_loss_avg)
            history['train_accuracy'].append(train_acc)
            history['val_accuracy'].append(val_acc)
            history['transition_accuracy'].append(trans_acc)

            # Learning rate scheduling
            if HAS_TRAINING_UTILS:
                scheduler.step()
            else:
                scheduler.step(val_loss_avg)

            # Early stopping
            if val_loss_avg < best_val_loss:
                best_val_loss = val_loss_avg
                patience_counter = 0
                torch.save(
                    model.state_dict(),
                    MODEL_DIR / 'tactical_state_predictor_best.pt'
                )
            else:
                patience_counter += 1

            # Logging
            if epoch % 10 == 0:
                marker = '*' if patience_counter == 0 else ''
                print(f"Epoch {epoch:3d}: train_loss={train_loss_avg:.4f}, "
                      f"val_loss={val_loss_avg:.4f}, "
                      f"state_acc={val_acc:.3f}, trans_acc={trans_acc:.3f} {marker}")

            if patience_counter >= config.patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

        print("-" * 70)
        print(f"Training complete. Best val loss: {best_val_loss:.4f}")

        return history


# =============================================================================
# INTEGRATION WITH EXISTING MODELS
# =============================================================================

def create_tactical_predictor_from_han(
    han_model: Optional['HierarchicalAttentionNetwork'] = None,
    config: Optional[TacticalPredictionConfig] = None
) -> 'TacticalStatePredictor':
    """
    Create a TacticalStatePredictor that uses HAN embeddings as features.

    Args:
        han_model: Optional pre-trained HAN model (for feature extraction)
        config: Prediction configuration

    Returns:
        Configured TacticalStatePredictor
    """
    if config is None:
        config = TacticalPredictionConfig()

    # Determine feature dimension
    if han_model is not None and HAS_HAN:
        # Use HAN's fused embedding dimension
        feature_dim = han_model.d_model
    else:
        # Use total raw features
        feature_dim = TOTAL_FEATURES if HAS_HAN else 198

    predictor = TacticalStatePredictor(
        feature_dim=feature_dim,
        config=config
    )

    return predictor


def get_state_description(state_idx: int, state_names: List[str] = None) -> Dict[str, Any]:
    """
    Get detailed description for a tactical state.

    Args:
        state_idx: State index
        state_names: Optional custom state names

    Returns:
        Dictionary with state information
    """
    if state_names is None:
        state_names = TacticalStateConfig().state_names

    if state_idx >= len(state_names):
        return {'error': f'Invalid state index: {state_idx}'}

    state_name = state_names[state_idx]

    if state_name in TACTICAL_STATE_DEFINITIONS:
        return {
            'name': state_name,
            'index': state_idx,
            **TACTICAL_STATE_DEFINITIONS[state_name]
        }
    else:
        return {
            'name': state_name,
            'index': state_idx,
            'description': 'No detailed description available'
        }


# =============================================================================
# MAIN / DEMONSTRATION
# =============================================================================

def print_architecture_summary():
    """Print summary of the tactical state prediction architecture."""

    print("=" * 80)
    print("TACTICAL STATE PREDICTION SYSTEM")
    print("S(t) -> S(t+1) Discrete State Transition Modeling")
    print("=" * 80)

    print("\n" + "-" * 80)
    print("TACTICAL STATES")
    print("-" * 80)

    config = TacticalStateConfig()
    for i, name in enumerate(config.state_names):
        desc = TACTICAL_STATE_DEFINITIONS.get(name, {})
        print(f"\n  {i}. {name.upper()}")
        if desc:
            print(f"     Description: {desc.get('description', 'N/A')}")
            print(f"     Indicators: {', '.join(desc.get('indicators', []))}")
            print(f"     Typical duration: {desc.get('typical_duration_days', 'N/A')} days")

    print("\n" + "-" * 80)
    print("ARCHITECTURE")
    print("-" * 80)
    print("""
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │                           INPUT FEATURES                                     │
    │              [batch, seq_len, feature_dim] from HAN or raw data             │
    └─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │                      TACTICAL STATE ENCODER                                  │
    │                                                                              │
    │   ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐     │
    │   │  Feature Encoder │ -> │ State Prototypes │ -> │  Gumbel-Softmax  │     │
    │   │  (MLP + LayerNorm)│    │  (Learnable)     │    │  (Soft/Hard)     │     │
    │   └──────────────────┘    └──────────────────┘    └──────────────────┘     │
    │                                                                              │
    │   Output: S(t) distribution [batch, n_states]                               │
    │           State embedding [batch, state_embed_dim]                          │
    └─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │                      STATE TRANSITION MODEL                                  │
    │                                                                              │
    │   ┌────────────────────────────────────────────────────────────────────┐   │
    │   │                    HYBRID TRANSITION                                │   │
    │   │                                                                     │   │
    │   │   ┌──────────────────┐         ┌──────────────────┐               │   │
    │   │   │  Markov Prior    │   +     │  Neural (Context) │               │   │
    │   │   │  (Learned Matrix)│  alpha  │  (MLP + Attention)│               │   │
    │   │   │                  │         │                   │               │   │
    │   │   │  P(S'|S) fixed   │         │  P(S'|S,context)  │               │   │
    │   │   └──────────────────┘         └──────────────────┘               │   │
    │   │                                                                     │   │
    │   │   alpha = mixing weight (learned based on context quality)         │   │
    │   └────────────────────────────────────────────────────────────────────┘   │
    │                                                                              │
    │   Output: P(S(t+1) | S(t), context) [batch, n_states]                       │
    └─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │                      UNCERTAINTY ESTIMATION                                  │
    │                                                                              │
    │   Separate network predicts epistemic uncertainty per state                 │
    │   Helps identify when predictions may be unreliable                         │
    │                                                                              │
    │   Output: uncertainty [batch, n_states]                                     │
    └─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │                         TRAJECTORY PREDICTION                                │
    │                                                                              │
    │   Multi-step: S(t) -> S(t+1) -> S(t+2) -> ... -> S(t+H)                     │
    │                                                                              │
    │   Uses transition model iteratively with uncertainty propagation            │
    └─────────────────────────────────────────────────────────────────────────────┘
    """)

    print("\n" + "-" * 80)
    print("LOSS FUNCTION COMPONENTS")
    print("-" * 80)
    print("""
    1. STATE CLASSIFICATION LOSS
       - Cross-entropy for current state prediction
       - Label smoothing for better generalization

    2. TRANSITION KL LOSS
       - KL divergence between predicted and actual next state
       - Measures transition prediction quality

    3. TEMPORAL CONSISTENCY LOSS
       - Penalizes rapid state changes
       - Encourages smooth transitions

    4. UNCERTAINTY CALIBRATION LOSS
       - Uncertainty should correlate with prediction error
       - Enables reliable confidence estimates
    """)

    if HAS_TORCH:
        # Create model and show parameter count
        config = TacticalPredictionConfig()
        model = TacticalStatePredictor(feature_dim=198, config=config)

        n_params = sum(p.numel() for p in model.parameters())
        print(f"\n" + "-" * 80)
        print(f"MODEL PARAMETERS: {n_params:,}")
        print("-" * 80)

        # Component breakdown
        encoder_params = sum(p.numel() for p in model.state_encoder.parameters())
        transition_params = sum(p.numel() for p in model.transition_model.parameters())
        uncertainty_params = sum(p.numel() for p in model.uncertainty_head.parameters())

        print(f"  State Encoder: {encoder_params:,}")
        print(f"  Transition Model: {transition_params:,}")
        print(f"  Uncertainty Head: {uncertainty_params:,}")


def demo_prediction():
    """Demonstrate tactical state prediction with synthetic data."""

    if not HAS_TORCH:
        print("PyTorch required for demonstration")
        return

    print("\n" + "=" * 80)
    print("DEMONSTRATION: Tactical State Prediction")
    print("=" * 80)

    # Create model
    config = TacticalPredictionConfig(
        hidden_dim=64,
        state_config=TacticalStateConfig(state_embed_dim=32)
    )
    model = TacticalStatePredictor(feature_dim=198, config=config)

    # Synthetic input
    batch_size = 4
    seq_len = 8
    features = torch.randn(batch_size, seq_len, 198)

    print("\nInput shape:", features.shape)

    # Sequence prediction
    model.eval()
    with torch.no_grad():
        outputs = model.predict_sequence(features)

    print("\nSequence Prediction Outputs:")
    print(f"  State sequence shape: {outputs['state_sequence'].shape}")
    print(f"  Current state shape: {outputs['current_state'].shape}")
    print(f"  Next state shape: {outputs['next_state'].shape}")
    print(f"  Uncertainty shape: {outputs['uncertainty'].shape}")

    # Most likely states
    current_states = outputs['current_state'].argmax(dim=-1)
    next_states = outputs['next_state'].argmax(dim=-1)

    print("\nPredicted States (batch 0):")
    print(f"  Current: {config.state_config.state_names[current_states[0]]}")
    print(f"  Next: {config.state_config.state_names[next_states[0]]}")

    # Trajectory prediction
    with torch.no_grad():
        trajectory = model.predict_trajectory(features, horizon=7)

    print(f"\nTrajectory Prediction (7-step horizon):")
    print(f"  Trajectory shape: {trajectory['trajectory'].shape}")

    # Show trajectory for first sample
    traj_states = trajectory['trajectory'][0].argmax(dim=-1)
    print(f"  Predicted trajectory: ", end="")
    print(" -> ".join([config.state_config.state_names[s][:8] for s in traj_states]))

    # Show transition matrix
    print("\nLearned Base Transition Matrix:")
    trans_matrix = model.transition_model.get_transition_matrix().detach().numpy()
    print("  (rows = from state, cols = to state)")
    for i, name in enumerate(config.state_config.state_names):
        probs = trans_matrix[i]
        top_3 = np.argsort(probs)[-3:][::-1]
        print(f"  {name[:15]:<15}: ", end="")
        for j in top_3:
            to_name = config.state_config.state_names[j][:8]
            print(f"{to_name}={probs[j]:.2f} ", end="")
        print()


if __name__ == "__main__":
    print_architecture_summary()

    if HAS_TORCH:
        demo_prediction()
