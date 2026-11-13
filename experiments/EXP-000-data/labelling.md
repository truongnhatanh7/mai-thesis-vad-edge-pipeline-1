# UCF-Crime test-set lighting labels: instructions for the owner

File to fill in: `EXP-000-data/labelling_sheet.csv` (290 rows). Fill only `owner_label`, plus `owner_note` if you
want. Do not edit the other columns.

## Allowed values for `owner_label`

| Value | Use when |
|---|---|
| `day` | Natural daylight or a well-lit indoor scene, in colour. You can see faces and clothing colours easily. |
| `night` | Dark scene **in colour**: outdoor at night under street lights, or a dim indoor scene. Important detail is hard to see. |
| `ir` | **Greyscale night-mode** (infrared) footage. The image is black-and-white, often with bright faces and IR glare. |
| `unclear` | Mixed lighting (e.g. the lights switch on, or the camera moves between scenes), dawn/dusk, or you cannot decide. Explain in `owner_note`. |

Rules:
1. Label the **whole video** by the lighting during **most of its duration**. If the anomalous interval has different
   lighting from the rest, label `unclear` and write "anomaly part = night" (or similar) in `owner_note`.
2. **Greyscale daytime footage** (an old black-and-white camera in daylight) is `day`, not `ir`. Add the note
   "grey daytime".
3. Judge by **eye**, not from the `mean_luma` / `greyscale_frac` columns. Those columns are there to help you find
   videos quickly. The point of the labelling is to test, later, how well those numbers agree with your judgement.
   Copying them would make that test circular.
4. You can sort by `mean_luma` to work faster, but look at the video for every label.

## How to look at a video quickly

Open `/Users/atruong/thesis-data/ucf-crime/UCF_Crimes/Videos/<relpath>` in QuickTime or IINA. The relpath is in
`index_test.csv`. Scrub to about 3 points (start, middle, end). Aim for about 20–30 s per video, which is about
2–3 h in total.

Content warning: the videos are real crime CCTV, some of it violent (shootings, assaults).

## After labelling

Tell the agent. The next step pre-registers a luma rule (threshold chosen **before** comparing it with your labels,
or chosen on a held-out half) and reports how well it agrees with your labels (confusion matrix, Cohen's κ).
