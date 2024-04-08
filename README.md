To train the model we need to follow further steps:

1) To fine-tune the model we can use either one of the present datasets, or prepare our own one.
For the project, I have decided to use mozilla-foundation/common_voice_13_0, that is present on huggingface ==> https://huggingface.co/datasets/mozilla-foundation/common_voice_13_0.
It is done with the idea in our mind that here all the 3 needed languages are present (Turkish, Azerbaijani, and Swahili).

It's step #1 in main.py file.

2) Afterwards we upload pre-trained model ==> facebook/wav2vec2-large-xlsr-53, which is trained on a wide range of languages,
including Turkish, Azerbaijani, and Swahili.

3) Then we prepare the model for fine-tuning.

4) Actual fine-tuning is step #4 (mentioned in main.py).

5) Our fifth step is --> training our model.

6) And final step is -- testing (test_size=0.2, as usual).
