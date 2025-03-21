from nlp.prosody_prediction.eval_interface import ProsodyPredictionInterface

if __name__ == "__main__":
    ckpt_path = "/src/experiments/09_Oct_2023_16_47_roberta_multiclass_both/_checkpoints/epoch=14-step=7034.ckpt"
    device = "cpu"
    text = (
        "Jane, I don’t like cavillers or questioners; besides, there is something truly "
        "forbidding in a child taking up her elders in that manner."
    )

    interface = ProsodyPredictionInterface(ckpt_path=ckpt_path, device=device, lang="EN")
    text_with_prosody = interface.predict(text)
    for sent_id, sent in enumerate(text_with_prosody.sents):
        print(f"Sentence {sent_id}:")
        for token in sent.tokens:
            print(f"{token.text} -- {token.prosody}")

        print("***************************\n")
