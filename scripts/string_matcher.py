# Licensed under CC BY-NC-ND 4.0 — https://creativecommons.org/licenses/by-nc-nd/4.0/
import re
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

class StringAgentMatcher:
    def __init__(self, agent_dict, config):
        self.agent_dict = agent_dict
        self.config = config
        # Precompile regex patterns for exact matching (word boundaries)
        self.patterns = {}
        for canonical, synonyms in agent_dict.items():
            for syn in synonyms:
                # Escape regex, add word boundaries
                pattern = r'\b' + re.escape(syn.lower()) + r'\b'
                self.patterns[pattern] = canonical

    def match(self, text):
        text_norm = text.lower()
        matches = set()
        str_pattern = set()
        # Exact matching
        for pattern, canonical in self.patterns.items():
            if re.search(pattern, text_norm):
                matches.add(canonical)
                str_pattern.add(pattern)
        # Fuzzy matching (if enabled)
        if self.config.get("fuzzy", {}).get("enabled", False):
            threshold = self.config["fuzzy"]["threshold"]
            min_len = self.config["fuzzy"]["min_word_len"]
            words = set(text_norm.split())
            for word in words:
                if len(word) < min_len:
                    continue
                for canonical, synonyms in self.agent_dict.items():
                    for syn in synonyms:
                        sim = JaroWinkler.similarity(word, syn.lower())
                        # print(f"Comparing '{word}' to '{syn.lower()}': similarity={sim:.2f}")
                        if sim >= threshold:
                            matches.add(canonical)
        return list(matches),list(str_pattern)