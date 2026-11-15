"""python -m computation_core.assistant.train_nlu  -> trains the intent model and saves data/assistant_nlu.npz"""
import time

import numpy as np

from .eval_set import COMPARE_TEST, EVAL, HUMAN_TEST, FINAL, FINAL2, FINAL3, FINAL4, TOPIC_TEST
from .nlu_corpus import INTENTS, generate
from .nlu_grammar import TOPICS
from .nlu_model import MODEL_PATH, IntentEnsemble, IntentModel


def main(n_per_intent=900, epochs=12, n_models=5):
    labels = list(INTENTS)
    models = []
    for seed in range(n_models):
        X, y, T = generate(n_per_intent, seed)  # each member sees its own sampling of the corpus
        order = np.random.default_rng(seed).permutation(len(X))
        cut = int(0.9 * len(X))
        tr, va = order[:cut], order[cut:]
        m = IntentModel(labels, TOPICS, seed=seed)
        t = time.time()
        m.fit([X[i] for i in tr], [y[i] for i in tr], [T[i] for i in tr], epochs=epochs, verbose=False)
        m.calibrate([X[i] for i in va], [y[i] for i in va], [T[i] for i in va])
        print(f"member {seed}: {time.time() - t:.0f}s, held-out template acc intent {m.accuracy([X[i] for i in va], [y[i] for i in va]):.3f}, "
              f"topic {m.topic_accuracy([X[i] for i in va], [T[i] for i in va]):.3f}", flush=True)
        models.append(m)
    ens = IntentEnsemble(models)
    from .dialog import COMMAND, Assistant  # the answer the user experiences = the model plus its guards and the command rule
    from .entities import extract
    from .text import normalize
    bot = Assistant(None, {}, ens)

    def routed(text):
        intent = bot.route(text)[0]
        e = extract(text)
        if intent == "change_context" and (e.vehicles or e.sites or e.coords or e.dates) and not COMMAND.search(normalize(text)):
            intent = "try_option"
        return intent

    for name, S in (("EVAL (dev)", EVAL), ("FINAL (dev)", FINAL), ("FINAL2 (dev)", FINAL2), ("FINAL3 (dev, seen once)", FINAL3), ("FINAL4 (UNSEEN)", FINAL4), ("COMPARE_TEST (UNSEEN)", COMPARE_TEST), ("HUMAN_TEST (UNSEEN, how people really write)", HUMAN_TEST)):
        r = [routed(e[0]) for e in S]
        ok, cl = sum(i == e[1] for i, e in zip(r, S)), sum(i == "clarify" for i in r)
        print(f"{name}: raw model {ens.accuracy([e[0] for e in S], [e[1] for e in S]):.3f} | with guards: correct {ok / len(S):.1%}, asks {cl / len(S):.1%}, confidently wrong {(len(S) - ok - cl) / len(S):.1%} (n={len(S)})")
    tx, tt = [e[0] for e in TOPIC_TEST], [e[1] for e in TOPIC_TEST]
    is_improve = float(np.mean([ens.predict(x)[0] == "improve" for x in tx]))
    print(f"TOPIC_TEST ('what can I do to improve X', {len(tx)} unseen sentences): intent=improve {is_improve:.3f}, topic correct {ens.topic_accuracy(tx, tt):.3f}")
    ens.save()
    print("saved", len(models), "models next to", MODEL_PATH)
    return ens


if __name__ == "__main__":
    main()
