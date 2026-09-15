import torch
import torch.nn as nn

from src.models.query_attention import (
    AgentEntityHistoryEncoder,
    AgentQueryGenerator,
    DistanceAwareAttention,
)
from src.models.communication_agent import (
    CommunicationEnhancedAgent,
)
from src.models.belief_state import (
    LocalBeliefState,
    ReliabilityGatedFusion,
)


class MATCHAgent(nn.Module):
    """
    Complete communication-enhanced MATCH agent.

    Forward process:

        local entity observations
                    |
                    v
        AgentEntityHistoryEncoder
                    |
                    v
              agent hidden h_i
                    |
                    v
          AgentQueryGenerator
                    |
                    v
                query q_i
                    |
                    v
        DistanceAwareAttention
                    |
                    v
          personalized message m_i
                    |
                    v
        CommunicationEnhancedAgent
                    |
                    v
             action Q-values

    The Information Hub reconstruction module remains
    outside this class. This class receives reconstructed
    entity representations from the Hub.
    """

    def __init__(
        self,
        n_agents: int = 5,
        n_entities: int = 10,
        entity_observation_dim: int = 8,
        hidden_dim: int = 32,
        query_dim: int = 32,
        reconstructed_entity_dim: int = 32,
        n_actions: int = 5,
        n_heads: int = 4,
        n_transformer_layers: int = 2,
        use_local_belief: bool = False,
        belief_hidden_dim: int = 32,
        belief_state_dim: int = 8,
        belief_logvar_min: float = -6.0,
        belief_logvar_max: float = 2.0,
        fuse_local_belief: bool = True,
        use_belief_history: bool = True,
    ):
        super().__init__()

        self.n_agents = n_agents
        self.n_entities = n_entities
        self.entity_observation_dim = (
            entity_observation_dim
        )
        self.hidden_dim = hidden_dim
        self.query_dim = query_dim
        self.reconstructed_entity_dim = (
            reconstructed_entity_dim
        )
        self.n_actions = n_actions
        self.use_local_belief = use_local_belief
        self.belief_hidden_dim = belief_hidden_dim
        self.fuse_local_belief = fuse_local_belief
        self.use_belief_history = use_belief_history

        if not self.use_local_belief and (
            not self.fuse_local_belief
            or not self.use_belief_history
        ):
            raise ValueError(
                "Belief ablations require use_local_belief=True."
            )

        self.history_encoder = (
            AgentEntityHistoryEncoder(
                n_agents=n_agents,
                n_entities=n_entities,
                entity_observation_dim=(
                    entity_observation_dim
                ),
                hidden_dim=hidden_dim,
                n_heads=n_heads,
                n_layers=n_transformer_layers,
            )
        )

        self.query_generator = AgentQueryGenerator(
            hidden_dim=hidden_dim,
            query_dim=query_dim,
        )

        self.distance_attention = (
            DistanceAwareAttention(
                query_dim=query_dim,
                entity_dim=(
                    reconstructed_entity_dim
                ),
            )
        )

        self.q_network = (
            CommunicationEnhancedAgent(
                hidden_dim=hidden_dim,
                message_dim=(
                    reconstructed_entity_dim
                ),
                n_actions=n_actions,
            )
        )

        if self.use_local_belief:
            self.local_belief = LocalBeliefState(
                n_agents=n_agents,
                n_entities=n_entities,
                entity_observation_dim=entity_observation_dim,
                agent_hidden_dim=hidden_dim,
                belief_hidden_dim=belief_hidden_dim,
                belief_state_dim=belief_state_dim,
                message_dim=reconstructed_entity_dim,
                logvar_min=belief_logvar_min,
                logvar_max=belief_logvar_max,
            )
            self.belief_fusion = ReliabilityGatedFusion(
                message_dim=reconstructed_entity_dim,
            )


    def initial_hidden(
        self,
        batch_size: int,
        device=None,
    ):
        """
        Create the initial historical hidden states.

        Output:
            [batch_size, n_agents, hidden_dim]
        """

        if device is None:
            device = next(
                self.parameters()
            ).device

        return self.history_encoder.initial_hidden(
            batch_size=batch_size,
            device=device,
        )

    def initial_belief_hidden(
        self,
        batch_size: int,
        device=None,
    ):
        if not self.use_local_belief:
            return None

        if device is None:
            device = next(self.parameters()).device

        return self.local_belief.initial_hidden(
            batch_size=batch_size,
            device=device,
        )

    def forward(
        self,
        entity_observations,
        observation_mask,
        previous_hidden,
        reconstructed_entities,
        reconstructed_state,
        entity_mask=None,
        previous_belief_hidden=None,
        hub_reliability=None,
    ):
        """
        Args:
            entity_observations:
                Local entity observations.

                Shape:
                    [batch_size,
                     n_agents,
                     n_entities,
                     entity_observation_dim]

            observation_mask:
                Whether each entity is visible to each agent.

                Shape:
                    [batch_size,
                     n_agents,
                     n_entities]

            previous_hidden:
                Historical hidden state from the previous
                timestep.

                Shape:
                    [batch_size,
                     n_agents,
                     hidden_dim]

            reconstructed_entities:
                Reconstructed entity embeddings produced
                by the Information Hub.

                Shape:
                    [batch_size,
                     n_entities,
                     reconstructed_entity_dim]

            reconstructed_state:
                Reconstructed entity state. The first two
                features are entity positions. The first
                n_agents entities represent the agents.

                Shape:
                    [batch_size,
                     n_entities,
                     state_entity_dim]

            entity_mask:
                Optional validity mask for reconstructed
                entities.

                Shape:
                    [batch_size, n_entities]

        Returns:
            Dictionary containing Q-values and all
            intermediate representations.
        """

        if entity_observations.ndim != 4:
            raise ValueError(
                "entity_observations must have shape "
                "[batch, agents, entities, features]."
            )

        if observation_mask.ndim != 3:
            raise ValueError(
                "observation_mask must have shape "
                "[batch, agents, entities]."
            )

        if reconstructed_entities.ndim != 3:
            raise ValueError(
                "reconstructed_entities must have shape "
                "[batch, entities, features]."
            )

        if reconstructed_state.ndim != 3:
            raise ValueError(
                "reconstructed_state must have shape "
                "[batch, entities, features]."
            )

        if reconstructed_state.shape[-1] < 2:
            raise ValueError(
                "reconstructed_state must contain at "
                "least two position features."
            )

        batch_size = entity_observations.shape[0]

        expected_observation_shape = (
            batch_size,
            self.n_agents,
            self.n_entities,
        )

        if (
            entity_observations.shape[:3]
            != expected_observation_shape
        ):
            raise ValueError(
                "Unexpected entity observation shape."
            )

        if (
            observation_mask.shape
            != expected_observation_shape
        ):
            raise ValueError(
                "observation_mask does not match the "
                "entity observations."
            )

        if (
            previous_hidden.shape
            != (
                batch_size,
                self.n_agents,
                self.hidden_dim,
            )
        ):
            raise ValueError(
                "Unexpected previous_hidden shape."
            )

        if reconstructed_entities.shape[:2] != (
            batch_size,
            self.n_entities,
        ):
            raise ValueError(
                "Unexpected reconstructed_entities "
                "shape."
            )

        if (
            reconstructed_entities.shape[-1]
            != self.reconstructed_entity_dim
        ):
            raise ValueError(
                "Unexpected reconstructed entity "
                "feature dimension."
            )

        if reconstructed_state.shape[:2] != (
            batch_size,
            self.n_entities,
        ):
            raise ValueError(
                "Unexpected reconstructed_state shape."
            )

        # Encode the current local entity observations
        # together with the previous historical state.
        agent_hidden = self.history_encoder(
            entity_observations=(
                entity_observations
            ),
            observation_mask=observation_mask,
            previous_hidden=previous_hidden,
        )

        # Generate each agent's query.
        queries = self.query_generator(
            agent_hidden
        )

        # Entity IDs 0,...,n_agents-1 correspond to agents.
        agent_positions = reconstructed_state[
            :,
            :self.n_agents,
            0:2,
        ]

        entity_positions = reconstructed_state[
            :,
            :,
            0:2,
        ]

        attention_result = (
            self.distance_attention(
                queries=queries,
                entity_representations=(
                    reconstructed_entities
                ),
                agent_positions=agent_positions,
                entity_positions=entity_positions,
                entity_mask=entity_mask,
            )
        )

        hub_messages = attention_result[
            "messages"
        ]

        belief_output = None
        decision_message = hub_messages

        if self.use_local_belief:
            if previous_belief_hidden is None:
                raise ValueError(
                    "previous_belief_hidden is required when "
                    "use_local_belief=True."
                )
            if hub_reliability is None:
                raise ValueError(
                    "hub_reliability is required when "
                    "use_local_belief=True."
                )

            belief_recurrent_state = previous_belief_hidden
            if not self.use_belief_history:
                # Remove only the recurrence introduced by DB-MATCH.  The
                # original MATCH agent history and Hub history remain intact,
                # making this a controlled Local-Belief-history ablation.
                belief_recurrent_state = torch.zeros_like(
                    previous_belief_hidden
                )

            belief_output = self.local_belief(
                entity_observations=entity_observations,
                observation_mask=observation_mask,
                agent_hidden=agent_hidden,
                previous_belief_hidden=belief_recurrent_state,
            )
            if self.fuse_local_belief:
                decision_message, belief_gate = self.belief_fusion(
                    hub_message=hub_messages,
                    local_message=belief_output["local_message"],
                    hub_reliability=hub_reliability,
                    mean_variance=belief_output["mean_variance"],
                )
            else:
                # Keep and supervise the Local Belief branch, but prevent its
                # message from affecting action selection.  A gate value of
                # one denotes the actual Hub-only decision path.
                decision_message = hub_messages
                belief_gate = torch.ones(
                    *hub_messages.shape[:-1],
                    1,
                    dtype=hub_messages.dtype,
                    device=hub_messages.device,
                )
            belief_output["gate"] = belief_gate

        # Combine historical information with the Hub/local-belief message.
        q_values = self.q_network(
            agent_hidden=agent_hidden,
            hub_messages=decision_message,
        )

        return {
            "q_values": q_values,
            "hidden_states": agent_hidden,
            "queries": queries,
            "hub_messages": hub_messages,
            "decision_messages": decision_message,
            "belief_hidden": (
                None
                if belief_output is None
                else belief_output["hidden"]
            ),
            "belief_mean": (
                None
                if belief_output is None
                else belief_output["mean"]
            ),
            "belief_logvar": (
                None
                if belief_output is None
                else belief_output["logvar"]
            ),
            "belief_gate": (
                None
                if belief_output is None
                else belief_output["gate"]
            ),
            "attention_weights": attention_result[
                "attention_weights"
            ],
            "normalized_distances": (
                attention_result[
                    "normalized_distances"
                ]
            ),
            "locality_weights": attention_result[
                "locality_weights"
            ],
        }
