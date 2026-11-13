# Draft: data/code request to the corresponding author (FOR THE OWNER TO SEND, nothing sent)

The owner fills in:
- the corresponding author's address, from the article's "Author information" or "Correspondence" section on nature.com
- their own name, university and supervisor

---

**To:** <corresponding author, as listed on the article>
**Subject:** Request for dataset and code: "An embedded deep learning framework for real-time violence detection and alert generation" (Sci. Rep. 2026)

Dear Dr. <name>,

I am a master's student at <university>, working under the supervision of <supervisor>. My thesis is on efficient,
CPU-only video analytics for public-safety use on edge devices. I read your recent Scientific Reports article
(doi:10.1038/s41598-026-44939-x) with great interest. Its Raspberry Pi 5 deployment is very close to my own setting.

I am trying to reproduce the CNN–GRU model as a baseline. The article says that the custom six-class dataset is available
from the corresponding author upon reasonable request. I would be grateful if you could share:

1. The custom six-class dataset (or the list of source videos and annotations), for non-commercial academic research only.
2. If possible, the training code or the trained weights / ONNX model.
3. A few details I could not find in the text:
   - the clip length (number of frames per GRU sequence)
   - the convolution strides
   - how the train/validation/test splits were made for Hockey Fight, RLVS and Violent Flows. In particular, was the
     split done per video or per frame/clip?
   - how the 1,000 test samples in Table 9 were obtained for each dataset

I will cite your work, use the data only for this thesis, and not redistribute it. I am happy to sign a data use
agreement.

Thank you for your time.

Kind regards,
<name>
<programme, university>
<email>
