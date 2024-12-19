from typing import List

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

from red_teaming.utility_functions.metrics import Metrics


class GibberishScore(Metrics):
    def __init__(self, method: str = 'gibberish') -> None:
        super().__init__()
        self.name = 'GibberishScore'
        self.tokenizer = AutoTokenizer.from_pretrained("madhurjindal/autonlp-Gibberish-Detector-492513457")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "madhurjindal/autonlp-Gibberish-Detector-492513457", 
        )
        self.method = method
        self.id2label = self.model.config.id2label
        self.maxlen = self.tokenizer.max_len_single_sentence

        """
        Noise: Gibberish at the zero level where even the different constituents of the input phrase (words) do not hold any meaning independently.
        For example: dfdfer fgerfow2e0d qsqskdsd djksdnfkff swq.

        Word Salad: Gibberish at level 1 where words make sense independently, but when looked at the bigger picture (the phrase) any meaning is not depicted.
        For example: 22 madhur old punjab pickle chennai

        Mild gibberish: Gibberish at level 2 where there is a part of the sentence that has grammatical errors, word sense errors, or any syntactical abnormalities, which leads the sentence to miss out on a coherent meaning.
        For example: Madhur study in a teacher

        Clean: This category represents a set of words that form a complete and meaningful sentence on its own.
        For example: I love this website.
        """

    @torch.no_grad()
    def get_score(self, texts: List[str], **kwargs) -> torch.Tensor:
        tokens = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        output = self.model(**tokens.to(self.model.device))

        probs = torch.nn.functional.softmax(output.logits, dim=-1)

        labels = self.id2label

        score_dicts = [{labels[i]: score.item() for i, score in enumerate(prob)} for prob in probs.data]

        final_scores = []
        if self.method == 'gibberish': # max of noise and word salad
            for score_dict in score_dicts:
                final_scores.append(max(score_dict['noise'], score_dict['word salad']))
        elif self.method == 'max_without_clean': # max of noise, word salad, and mild gibberish
            for score_dict in score_dicts:
                final_scores.append(max(score_dict['noise'], score_dict['word salad'], score_dict['mild gibberish']))
        elif self.method == 'not_clean': # opposite of clean
            for score_dict in score_dicts:
                final_scores.append((1 - score_dict['clean']))
        elif self.method in ['noise', 'word salad', 'mild gibberish']: # chose the class (if choose clean)
            for score_dict in score_dicts:
                final_scores.append(score_dict[self.method])
        else :
            raise NotImplementedError
        return -torch.tensor(final_scores)
