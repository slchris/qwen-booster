"""Incrementally split Qwen thinking output; the opening tag is in the prompt."""


class ReasoningParser:
    def __init__(self, thinking=False):
        self.thinking = thinking
        self.pending = ""

    def feed(self, text, final=False):
        if not self.thinking:
            return {"content": text} if text else {}
        self.pending += text
        marker = "</think>"
        if marker in self.pending:
            reasoning, answer = self.pending.split(marker, 1)
            self.pending = ""
            self.thinking = False
            return {key: value for key, value in
                    (("reasoning_content", reasoning), ("content", answer)) if value}
        keep = 0
        if not final:
            for n in range(1, len(marker)):
                if self.pending.endswith(marker[:n]):
                    keep = n
        emitted = self.pending[:-keep] if keep else self.pending
        self.pending = self.pending[-keep:] if keep else ""
        return {"reasoning_content": emitted} if emitted else {}
