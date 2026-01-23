
from dataclasses import dataclass
from typing import Any

# Mocking the classes to verify property access

@dataclass
class MatchResult:
    best_profile: str | None
    confidence: float
    expected_duration: float
    matched_phase: str | None
    ranking: list[dict[str, Any]]
    is_ambiguous: bool
    margin: float
    is_confident_mismatch: bool = False

class Manager:
    def __init__(self):
        self._last_match_result = None
        
    @property
    def top_candidates(self) -> list[dict[str, Any]]:
        """Return the list of top candidates from the last match."""
        if self._last_match_result and hasattr(self._last_match_result, "ranking"):
            print(f"DEBUG: Found ranking in result: {self._last_match_result.ranking}")
            return self._last_match_result.ranking
        print("DEBUG: No result or ranking attribute")
        return []

# Test 1: Initially None
mgr = Manager()
print(f"Test 1 (Init): {mgr.top_candidates}")

# Test 2: Populated
res = MatchResult("Test", 0.9, 100, None, [{"name": "Test", "score": 0.9}], False, 0.1)
mgr._last_match_result = res
print(f"Test 2 (Populated): {mgr.top_candidates}")

# Test 3: Empty candidates
res_empty = MatchResult(None, 0.0, 0, None, [], False, 0.0)
mgr._last_match_result = res_empty
print(f"Test 3 (Empty): {mgr.top_candidates}")

# Test 4: Check if MatchResult actually matches what I see in profile_store.py
# In profile_store.py:
# return MatchResult(best["name"], ..., candidates, ...)
# Wait, currently MatchResult is defined in profile_store.py but I need to check its __init__
