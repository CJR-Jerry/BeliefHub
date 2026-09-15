import numpy as np


class SMACEntityObservationAdapter:
    """Build MATCH entity tensors from an active SMAC environment.

    Entities are ordered as all allied units followed by all enemy units.
    The first two features are positions so the existing MATCH losses and
    distance-aware attention keep their coordinate convention.

    Base feature layout (followed by SMAC unit-type one-hot bits):
        0:2  position (agent-relative locally, map-centred globally)
        2    normalized health
        3    normalized shield (zero on non-Protoss maps)
        4    ally indicator
        5    enemy indicator
        6    alive indicator
        7    reserved (zero; keeps the CN/MATCH eight-feature interface)
    """

    base_entity_dim = 8

    def __init__(self, env):
        self.env = env
        self.n_agents = env.n_agents
        self.n_enemies = env.n_enemies
        self.n_entities = self.n_agents + self.n_enemies
        self.unit_type_bits = env.unit_type_bits
        self.entity_observation_dim = (
            self.base_entity_dim + self.unit_type_bits
        )
        self.entity_state_dim = self.entity_observation_dim

    def _scale_x(self):
        return max(float(self.env.max_distance_x), 1.0)

    def _scale_y(self):
        return max(float(self.env.max_distance_y), 1.0)

    def _shield(self, unit):
        if not hasattr(unit, "shield") or unit.shield <= 0:
            return 0.0
        maximum = self.env.unit_max_shield(unit)
        if maximum is None or maximum <= 0:
            return 0.0
        return float(unit.shield / maximum)

    def _type_features(self, unit, ally):
        features = np.zeros(self.unit_type_bits, dtype=np.float32)
        if self.unit_type_bits == 0:
            return features
        type_id = int(self.env.get_unit_type_id(unit, ally))
        if not 0 <= type_id < self.unit_type_bits:
            raise ValueError(
                f"SMAC returned invalid unit type {type_id} for "
                f"{self.unit_type_bits} type bits."
            )
        features[type_id] = 1.0
        return features

    def _entity_features(self, unit, ally, x, y):
        features = np.zeros(
            self.entity_observation_dim,
            dtype=np.float32,
        )
        if unit.health <= 0:
            return features

        features[0] = float(x)
        features[1] = float(y)
        features[2] = float(unit.health / unit.health_max)
        features[3] = self._shield(unit)
        features[4] = float(ally)
        features[5] = float(not ally)
        features[6] = 1.0
        if self.unit_type_bits:
            features[8:] = self._type_features(unit, ally)
        return features

    def convert(self):
        """Return local entity observations and visibility masks.

        Raw unit objects are read only for units SMAC marks visible.  Thus a
        hidden unit contributes an all-zero vector and no privileged state is
        leaked into decentralized execution.
        """

        visibility = np.asarray(
            self.env.get_visibility_matrix(),
            dtype=bool,
        )
        expected_shape = (self.n_agents, self.n_entities)
        if visibility.shape != expected_shape:
            raise ValueError(
                f"visibility has shape {visibility.shape}, expected "
                f"{expected_shape}."
            )

        observations = np.zeros(
            (
                self.n_agents,
                self.n_entities,
                self.entity_observation_dim,
            ),
            dtype=np.float32,
        )

        # SMAC's visibility matrix intentionally omits the diagonal.  Each
        # living agent always observes itself.
        for agent_id in range(self.n_agents):
            if self.env.agents[agent_id].health > 0:
                visibility[agent_id, agent_id] = True

        for observer_id in range(self.n_agents):
            observer = self.env.agents[observer_id]
            if observer.health <= 0:
                visibility[observer_id, :] = False
                continue

            observer_x = float(observer.pos.x)
            observer_y = float(observer.pos.y)

            for ally_id, ally in self.env.agents.items():
                if visibility[observer_id, ally_id]:
                    observations[observer_id, ally_id] = (
                        self._entity_features(
                            ally,
                            ally=True,
                            x=(float(ally.pos.x) - observer_x) / self._scale_x(),
                            y=(float(ally.pos.y) - observer_y) / self._scale_y(),
                        )
                    )

            for enemy_id, enemy in self.env.enemies.items():
                entity_id = self.n_agents + enemy_id
                if visibility[observer_id, entity_id]:
                    observations[observer_id, entity_id] = (
                        self._entity_features(
                            enemy,
                            ally=False,
                            x=(float(enemy.pos.x) - observer_x) / self._scale_x(),
                            y=(float(enemy.pos.y) - observer_y) / self._scale_y(),
                        )
                    )

        return observations, visibility

    def build_global_state(self):
        """Return the privileged entity target used only by the learner."""

        state = np.zeros(
            (self.n_entities, self.entity_state_dim),
            dtype=np.float32,
        )
        center_x = float(self.env.map_x) / 2.0
        center_y = float(self.env.map_y) / 2.0

        for ally_id, ally in self.env.agents.items():
            state[ally_id] = self._entity_features(
                ally,
                ally=True,
                x=(float(ally.pos.x) - center_x) / self._scale_x(),
                y=(float(ally.pos.y) - center_y) / self._scale_y(),
            )

        for enemy_id, enemy in self.env.enemies.items():
            state[self.n_agents + enemy_id] = self._entity_features(
                enemy,
                ally=False,
                x=(float(enemy.pos.x) - center_x) / self._scale_x(),
                y=(float(enemy.pos.y) - center_y) / self._scale_y(),
            )

        return state
