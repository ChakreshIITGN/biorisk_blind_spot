# Licensed under CC BY-NC-ND 4.0 — https://creativecommons.org/licenses/by-nc-nd/4.0/
from sentence_transformers import SentenceTransformer, util

class SemanticAgentMatcher:
    def __init__(self, agent_dict, config):
        self.agent_dict = agent_dict
        self.config = config
        if config["method"] == "embedding":
            self.model = SentenceTransformer(config["embedding_model"])
            # Pre‑compute embeddings for all canonical agent names + synonyms
            self.agent_embeddings = {canon: self.model.encode(syn) for canon, syns in agent_dict.items() for syn in syns}
    def match(self, text):
        if self.config["method"] == "embedding":
            text_emb = self.model.encode(text)
            best_score = 0.0
            best_agent = None
            for agent, emb in self.agent_embeddings.items():
                sim = util.cos_sim(text_emb, emb).item()
                if sim > best_score and sim >= self.config["similarity_threshold"]:
                    best_score = sim
                    best_agent = agent
            return [best_agent] if best_agent else []
        elif self.config["method"] == "llm":
            # Call GPT‑4o‑mini with a prompt: "Does this text refer to any of these agents? Return JSON."
            # (Implementation omitted for brevity)
            pass