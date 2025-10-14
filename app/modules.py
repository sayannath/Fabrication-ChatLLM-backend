import dspy


class QA(dspy.Signature):
    """Signature for Answer Generation task."""

    question = dspy.InputField(desc="Question field")
    answer = dspy.OutputField(desc="Answer field")


class ChainOfThought(dspy.Module):
    """Custom module for chain-of-thought reasoning."""

    def __init__(self):
        self.cot = dspy.ChainOfThought(QA)

    def forward(self, question: str):
        return self.cot(question=question)
