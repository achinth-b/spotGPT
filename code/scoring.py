import os
import numpy as np
from dotenv import load_dotenv
import openai

# API KEYS
load_dotenv()


def openai_setup():
    openai.organization = "org-DAKfxuGuaoVJNaFffnWZ3Ueu"
    openai.api_key = os.getenv("OPENAI_API_KEY")
    (openai.Model.list())


def get_log_probability(text):
    kwargs = {"engine": 'davinci',
              "temperature": 0,
              "max_tokens": 0,
              "echo": True,
              "logprobs": 0}
    op = openai.Completion.create(prompt=f"<|endoftext|>{text}", **kwargs)
    result = op['choices'][0]
    tokens, logprobs = result["logprobs"]["tokens"][1:
                                                    ], result["logprobs"]["token_logprobs"][1:]

    assert len(tokens) == len(
        logprobs), f"Expected {len(tokens)} logprobs, got {len(logprobs)}"

    return np.mean(logprobs)


sample_text_mgt = "In Finnish folklore, all places and things, and also human beings, have a haltija (a genius, guardian spirit) of their own. One such haltija is called etiäinen—an image, doppelgänger, or just an impression that goes ahead of a person, doing things the person in question later does. For example, people waiting at home might hear the door close or even see a shadow or a silhouette, only to realize that no one has yet arrived. Etiäinen can also refer to some kind of a feeling that something is going to happen. Sometimes it could, for example, warn of a bad year coming. In modern Finnish, the term has detached from its shamanistic origins and refers to premonition. Unlike clairvoyance, divination, and similar practices, etiäiset (plural) are spontaneous and can't be induced. Quite the opposite, they may be unwanted and cause anxiety, like ghosts. Etiäiset need not be too dramatic and may concern everyday events, although ones related to e.g. deaths are common. As these phenomena are still reported today, they can be considered a living tradition, as a way to explain the psychological experience of premonition."

sample_text_hgt = "In Finnish folklore, all places and things, animate or inanimate, have a spirit or \"etiäinen\" that lives there. Etiäinen can manifest in many forms, but is usually described as a kind, elderly woman with white hair. She is the guardian of natural places and often helps people in need. Etiäinen has been a part of Finnish culture for centuries and is still widely believed in today. Folklorists study etiäinen to understand Finnish traditions and how they have changed over time."


openai_setup()
print("Log-probability of the machine generated text", get_log_probability(sample_text_mgt))
print("Log-probability of the human generated text", get_log_probability(sample_text_hgt))
