"""
Executor Module - Pairwise Engine

Date: 2025-11-13
Author: Rebirthli
Description: Pairwise combinatorial test case generation using PICT algorithm
"""

import logging
import hashlib
import pickle
from typing import Any, Dict, List, Optional, Sequence
from pathlib import Path

from ..config.utils.exceptions import CaseGenerationError

logger = logging.getLogger(__name__)


class PairwiseEngine:
    """Pairwise test case generation engine"""

    def __init__(
        self,
        engine_type: str = 'PICT',
        cache_dir: Optional[Path] = None
    ):
        """
        Initialize PairwiseEngine

        Args:
            engine_type: Algorithm type ('PICT', 'IPO', 'naive')
            cache_dir: Directory for caching results (optional)
        """
        self.engine_type = engine_type
        self.cache_dir = cache_dir
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        dimensions: Dict[str, Sequence[Any]],
        seed: Optional[int] = None,
        max_cases: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate pairwise test cases

        Args:
            dimensions: {param_name: [possible_values]}
            seed: Random seed for reproducibility
            max_cases: Maximum number of cases to generate

        Returns:
            List of test case parameter combinations

        Raises:
            CaseGenerationError: If generation fails
        """
        if not dimensions:
            return []

        # Check cache
        if self.cache_dir:
            cache_key = self._compute_cache_key(dimensions, seed)
            cached = self._load_from_cache(cache_key)
            if cached:
                logger.info(f"Loaded {len(cached)} cases from cache")
                return cached[:max_cases] if max_cases else cached

        # Handle high-dimensional cases
        if len(dimensions) > 10:
            logger.info(f"High dimensionality ({len(dimensions)}), using hierarchical strategy")
            combos = self._generate_hierarchical(dimensions, seed)
        else:
            # Use selected engine
            if self.engine_type == 'PICT':
                combos = self._generate_pict(dimensions, seed)
            elif self.engine_type == 'IPO':
                combos = self._generate_ipo(dimensions, seed)
            else:
                combos = self._generate_naive(dimensions, seed)

        # Cache results BEFORE applying max_cases limit
        if self.cache_dir:
            self._save_to_cache(cache_key, combos)

        # Apply max_cases limit AFTER caching
        if max_cases and len(combos) > max_cases:
            logger.warning(
                f"Generated {len(combos)} cases, limiting to {max_cases}"
            )
            # Use deterministic sampling based on seed
            if seed is not None:
                import random
                random.seed(seed)
                combos = random.sample(combos, max_cases)
            else:
                combos = combos[:max_cases]

        logger.info(f"Generated {len(combos)} pairwise test cases")
        return combos

    def _generate_pict(
        self,
        dimensions: Dict[str, Sequence[Any]],
        seed: Optional[int]
    ) -> List[Dict[str, Any]]:
        """Generate using PICT algorithm (via allpairspy)"""
        try:
            from allpairspy import AllPairs

            param_names = list(dimensions.keys())
            param_values = [list(dimensions[k]) for k in param_names]

            # Note: allpairspy doesn't support seed parameter
            pairs = AllPairs(param_values)

            result = []
            for combo in pairs:
                result.append(dict(zip(param_names, combo)))

            return result

        except ImportError:
            logger.warning("allpairspy not installed, falling back to naive generation")
            return self._generate_naive(dimensions, seed)
        except Exception as e:
            raise CaseGenerationError(f"PICT generation failed: {e}")

    def _generate_ipo(
        self,
        dimensions: Dict[str, Sequence[Any]],
        seed: Optional[int]
    ) -> List[Dict[str, Any]]:
        """Generate using IPO algorithm (In-Parameter-Order)"""
        # For simplicity, fall back to PICT as allpairspy supports IPO-like behavior
        return self._generate_pict(dimensions, seed)

    def _generate_naive(
        self,
        dimensions: Dict[str, Sequence[Any]],
        seed: Optional[int]
    ) -> List[Dict[str, Any]]:
        """Generate using naive cartesian product (limited to 100 cases)"""
        import itertools

        param_names = list(dimensions.keys())
        param_values = [list(dimensions[k]) for k in param_names]

        cartesian = itertools.product(*param_values)

        result = []
        for combo in itertools.islice(cartesian, 100):  # Limit to 100
            result.append(dict(zip(param_names, combo)))

        logger.warning(f"Naive generation limited to {len(result)} cases")
        return result

    def _generate_hierarchical(
        self,
        dimensions: Dict[str, Sequence[Any]],
        seed: Optional[int]
    ) -> List[Dict[str, Any]]:
        """
        Hierarchical strategy for high-dimensional cases

        Use pairwise on first 5 dimensions, use defaults for rest
        """
        # Select first 5 dimensions for pairwise
        priority_dims = dict(list(dimensions.items())[:5])
        default_dims = {k: list(v)[0] for k, v in list(dimensions.items())[5:]}

        # Generate pairwise for priority dimensions
        priority_combos = self._generate_pict(priority_dims, seed)

        # Merge with defaults
        full_combos = []
        for combo in priority_combos:
            full_combos.append({**combo, **default_dims})

        return full_combos

    def _compute_cache_key(
        self,
        dimensions: Dict[str, Sequence[Any]],
        seed: Optional[int]
    ) -> str:
        """Compute cache key for dimensions + seed"""
        data = {
            'dimensions': {k: list(v) for k, v in dimensions.items()},
            'seed': seed,
            'engine': self.engine_type
        }
        serialized = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
        return hashlib.md5(serialized).hexdigest()

    def _load_from_cache(self, cache_key: str) -> Optional[List[Dict[str, Any]]]:
        """Load cached results"""
        if not self.cache_dir:
            return None

        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
        return None

    def _save_to_cache(self, cache_key: str, combos: List[Dict[str, Any]]) -> None:
        """Save results to cache"""
        if not self.cache_dir:
            return

        cache_file = self.cache_dir / f"{cache_key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(combos, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.debug(f"Saved {len(combos)} cases to cache")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
