# Disaster Damage Assessment

Classify building damage from a pair of overhead images of the same place, one taken before an event and one taken after. The idea matches how analysts work after a storm or earthquake: a single picture rarely tells you whether a building is wrecked, but the difference between the two times does.

This repo builds a small siamese network that looks at the before image and the after image with one shared encoder, then decides from how the scene changed. It also includes a single image baseline that only sees the after picture, so we can prove the pair actually matters.

## Why a siamese model

A roof that looks like rubble in the after image might have been a parking lot or a textured field all along. The honest signal is the change. The siamese model runs the same convolutional encoder over both images and classifies from the shift in the embedding. Because the two branches share weights, the network is set up to measure difference rather than to memorize either frame.

The single image baseline reuses the same encoder and the same head size but only sees the after image. Any accuracy gap between the two comes from access to the pair, not from extra parameters.

## Synthetic data

There is no external download. The dataset is generated on the fly in `src/data.py`. Each sample starts from a base scene with a randomly placed bright building footprint on a textured ground. The before and after frames are two noisy views of that same scene with label independent nuisance changes added to both: a global illumination shift, small sensor noise, and a one pixel registration jitter.

For a damaged pair, the after frame also gets a planted damage signature on part of the building: a high frequency rubble texture plus a debris darkening. The brightness pull down and the brightness add up are tuned so that the average brightness of the after image is about the same for damaged and undamaged cases. That is deliberate. It keeps the label out of any single frame, so the only reliable way to recover it is to compare before against after.

## Models

`src/model.py` contains:

* `ConvEncoder`, a real two layer CNN with global average pooling that maps one image to an embedding. It trains from scratch, so the tests stay offline with no pretrained weights.
* `SiameseDamageNet`, the shared encoder run over both images, classifying from the after minus before embedding.
* `SingleImageDamageNet`, the ablation that encodes one image only.

## Results from a real run

Running `python -m src.demo` trains both models on 240 synthetic pairs and evaluates on 120 held out pairs:

```
siamese (uses both images) test accuracy: 0.817
single image (after only) test accuracy:  0.500
```

The siamese model lands well above the 0.5 chance line, while the after only model sits at chance, which is exactly what we expect when the label is hidden from any single frame. Numbers will move a little with the random seed, but the ordering holds.

## Tests

The test suite checks behavior, not fixed magic numbers:

* the generator produces balanced, in range pairs, and damaged pairs really do change more between frames
* the after image alone does not separate the classes by brightness
* the siamese branches share weights, and the output genuinely depends on the before image
* the after only model is invariant to the before image
* trained damage classification beats chance
* the siamese model beats the single image model by a clear margin, and the single image model stays near chance

Run them with:

```
python -m pytest tests/ -q
```

## Layout

```
src/data.py    synthetic before and after pair generator
src/model.py   shared encoder, siamese model, single image ablation
src/train.py   training and evaluation helpers
src/demo.py    end to end demo that prints both accuracies
tests/         pytest behavior tests
```

## Requirements

Python with numpy, torch, and pytest. See `requirements.txt`. Everything runs on CPU in a few seconds.
