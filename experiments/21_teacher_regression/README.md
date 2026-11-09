# 21 — Does SuperFlex's network just learn how chairs are decomposed?

**Question.** The same objective gives better decompositions from SuperFlex's prediction than from k-means. Suspicion: the
network learns from simple chairs how a chair splits, which gives a better start on hard ones.

**Setup** (`code/train.py`). Student: SuperFlex's architecture from random weights. Only loss: L1 on the pretrained
network's raw outputs, slot by slot on the slots it uses, plus BCE on which those are. No geometry, no augmentation, 600
epochs, Adam 3e-4 with one cycle. The 1355 test chairs split into 1084 mini-train and 271 mini-test, the director's chair
in mini-test, last epoch used. All fits plain, on the fixed occupancy, with 1000 steps of SuperFlex's objective. K-means
uses the teacher's count per chair. Thin = lowest quartile of occupancy over the test split, 57 of the 271.

    uv run python experiments/21_teacher_regression/code/train.py --name student
    uv run python experiments/21_teacher_regression/code/dashboard.py      -> http://127.0.0.1:7862

| mini-test IoU | all | thin | others | director's chair | SQs |
|---|---|---|---|---|---|
| k-means boxes | 0.290 | 0.134 | 0.332 | 0.062 | 7.8 |
| k-means, optimized | 0.727 | 0.559 | 0.772 | 0.293 | 7.8 |
| student after 8 epochs | 0.194 | 0.067 | 0.227 | 0.055 | 7.5 |
| student after 8 epochs, optimized | 0.719 | 0.527 | 0.770 | 0.717 | 7.5 |
| student | 0.402 | 0.199 | 0.456 | 0.241 | 7.5 |
| student, optimized | 0.754 | 0.572 | 0.803 | 0.706 | 7.5 |
| teacher | 0.646 | 0.448 | 0.699 | 0.430 | 7.8 |
| teacher, optimized | 0.794 | 0.665 | 0.829 | 0.749 | 7.8 |

- Student against k-means, optimized: +0.027 ± 0.014 (95 %), better on 61 % of chairs. Teacher against k-means: +0.067 ± 0.013, 77 %.
- The student gains less on thin chairs (+0.013) than on the others (+0.031). The teacher gains +0.106 and +0.057.
- The student does not imitate the teacher on held-out chairs: train loss 0.52, test loss bottoms at 1.31 around epoch
  225 and ends at 1.47. Test IoU stops rising near epoch 400.
- On the director's chair the 8-epoch student is as good as the final one, so that gain needs almost no learning.

**Limits.** 1084 chairs without augmentation, against the teacher's much larger training, so the gap to the teacher can be
data rather than the loss. IoU only, nothing compared visually. One split and one start.
