import random
import string
from typing import Dict, List, Tuple

CANARY_TEMPLATE = "NAME: {name} | EMAIL: {email} | PHONE: {phone} | CODE: {code}"


def _rand_name(rng: random.Random) -> str:
    first = rng.choice(["Li", "Zhang", "Wang", "Chen", "Liu", "Zhao", "Yang", "Huang"])
    last = rng.choice(["Wei", "Ming", "Hao", "Tao", "Xin", "Yu", "Jie", "Yan"])
    return f"{first} {last}"


def _rand_email(rng: random.Random) -> str:
    user = "".join(rng.choice(string.ascii_lowercase + string.digits) for _ in range(10))
    dom = rng.choice(["example.com", "mail.test", "corp.local"])
    return f"{user}@{dom}"


def _rand_phone(rng: random.Random) -> str:
    return "1" + "".join(rng.choice(string.digits) for _ in range(10))


def _rand_code(rng: random.Random, n: int = 24) -> str:
    alphabet = string.ascii_letters + string.digits
    return "".join(rng.choice(alphabet) for _ in range(n))


def make_canaries(num: int, seed: int = 0) -> List[Dict]:
    rng = random.Random(seed)
    out = []
    for i in range(num):
        canary = CANARY_TEMPLATE.format(
            name=_rand_name(rng),
            email=_rand_email(rng),
            phone=_rand_phone(rng),
            code=_rand_code(rng),
        )
        # Prefix for extraction: reveal partial info to force completion
        prefix = canary.split("| CODE:")[0].strip() + " | CODE:"
        out.append({"id": i, "canary": canary, "prefix": prefix})
    return out


def insert_canaries(
    texts: List[str], canaries: List[Dict], repeats: int, seed: int = 0
) -> Tuple[List[str], Dict]:
    """
    Insert each canary `repeats` times into the corpus at random positions.
    Return new_texts and meta (for later evaluation).
    """
    rng = random.Random(seed)
    new_texts = list(texts)
    insertion_plan = []
    for c in canaries:
        for repeat_idx in range(repeats):
            pos = rng.randrange(0, len(new_texts) + 1)
            new_texts.insert(pos, c["canary"])
            insertion_plan.append(
                {"canary_id": c["id"], "repeat_index": repeat_idx, "sampled_pos": pos}
            )

    canary_to_id = {c["canary"]: c["id"] for c in canaries}
    positions = []
    for idx, text in enumerate(new_texts):
        canary_id = canary_to_id.get(text)
        if canary_id is not None:
            positions.append({"canary_id": canary_id, "pos": idx})

    meta = {
        "seed": seed,
        "repeats": repeats,
        "num_canaries": len(canaries),
        "input_size": len(texts),
        "output_size": len(new_texts),
        "positions": positions,
        "insertion_plan": insertion_plan,
        "canaries": canaries,
    }
    return new_texts, meta
