from asent.data_classes import SpanPolarityOutput
from fastapi import FastAPI
import spacy
import asent
from typing import Union, List, Any
from pydantic import BaseModel
from spacy.matcher import PhraseMatcher

from rake_nltk import Rake

nlp = spacy.load("en_core_web_lg")
nlp.add_pipe("asent_en_v1")
matcher = PhraseMatcher(nlp.vocab, attr="LOWER")

terms = ["quality", "pricing", "looks", "worth it"]
patterns = [nlp.make_doc(text) for text in terms]
matcher.add("TerminologyList", patterns)


def returnDocPolarity(self):
    return {"negative": round(self.negative, 3), "positive": round(self.positive, 3), "neutral": round(self.neutral, 3),
            "compound": round(self.compound, 3)}


def returnSentencePolarity(self):
    return {"negative": round(self.negative, 3), "positive": round(self.positive, 3),
            "neutral": round(self.neutral, 3),
            "compound": round(self.compound, 3), "span": str(self.span)}


class Text(BaseModel):
    reviews: List[str]


app = FastAPI()


@app.post("/results/")
async def create_item(Reviews: Text):
    result = []
    reviews = Reviews.reviews
    # analysis

    for single_review in reviews:
        review_result = {"review": single_review}

        # sentimental analysis
        doc = nlp(single_review)
        doc_polarity = doc._.polarity
        review_result["polarity"] = returnDocPolarity(doc_polarity)

        # named entities
        named_entities = {}
        for ent in doc.ents:
            named_entities[ent.text] = ent.label_
        review_result["entities"] = named_entities

        # Polarity of each sentence
        review_result["polarity_sentence"] = []
        sentence_dict = []
        for sentence in doc.sents:
            polarity_sentence = returnSentencePolarity(sentence._.polarity)
            sentence_dict.append(polarity_sentence)
        review_result["polarity_sentence"] = sentence_dict

        # Keywords Extracted
        r = Rake()
        r.extract_keywords_from_text(single_review)
        keywords = []
        [keywords.append(q) if len(q) > 12 else None for q in r.get_ranked_phrases()]
        review_result["keywords"] = keywords

        result.append(review_result)

    return result
