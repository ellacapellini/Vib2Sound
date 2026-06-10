### Literature Tracking Spreadsheet
[Link to Google Sheet Literature Review Papers](https://docs.google.com/spreadsheets/d/1AIvS_MhnGrPWRKrOBHwTzvh15Oux9gk9Hg6x9rD6eRY/edit?usp=sharing)
All paper notes, reading status, and annotations live in the spreadsheet above. 
Each paper is tagged with:
    🐦 bioacoustics 
    🎙️ speech separation 
    👁️ audio-visual 
    🔧 system/hardware 
    🧠 architecture 
    ❓ missing modality 
    ⭐ core reference
    🧱 theoretical foundation 
    📖 review 
    ⚠️ preprint

# PROJECT NOTES 
Vib2Sound assumes the acceler. is always there and clean. When it isn't — whether fully missing or just degraded — the model has nothing to fall back on. The *ablation* already showed the microphone array spatial info doesn't help, so there's really no acoustic rescue built in.
## Framing: a sort of Degradation Spectrum 
Most of the robustness literature treats missingness as binary (cue present or cue absent). But that's not what actually happens in BirdPark. Wing flaps, poor mechanical coupling, PLL failures, battery drift — these don't switch the accelerometer off, they degrade it gradually. The signal gets noisier, weaker, or corrupted in specific frequency bands.
So I think the more honest evaluation framework isn't a single missing/present test but a spectrum: fully observed → soft degradation (noise injection, gain reduction, band masking) → fully absent. 
## Four Complementary Approaches 
These aren't alternatives, they stack. CMC comes first — it shapes the audio representation so that everything else starts from a better base. Continuous degradation training is the evaluation and training design that stress-tests the model across the full spectrum. The cascade sits as "jolly" as the experimental control that makes the contribution of the accelerometer legible at each point. Cross-modal prediction would eventually sit alongside CMC as a more active version of the same intuition — but that's later.
### 1. CMC loss
Inspired by Makishima et al. (2021) → Add a *contrastive loss* during training that pulls matching bird/audio pairs together in emebedding space and pushes mismatched pairs apart. The ideas is that this forces audio encoder to learn identity-discriminative representations even without the acceler. explicitly present. It costs nothing at inference, the contrastive block is removed after training. it aslo potentially lifts the floor for the other approaches, since everything downstream benefits from a better audio representation. 
### 2. Cross-modal prediction 
(parking for now) the idea would be to train a module that predicts what the acceler. signal should look like from microphone input, and substitute that prediction when the real signal is degraded, conceptually related to the cross-modal generation literature and the sensor fusion framing in VibOmni (He et al. 2025), though neither addresses this exact setting. the honest uncertainty is whether audio-to-acceler. mapping is learnable at all given the ablation results. Worth revisiting once the others are running. 
### 3. Continuous Degradation Training
instead of randomly zeroing acceler. frames during training (binary dropout, as in Liao et al. 2019, and Chand et al. 2023), inject realistic corruptions (e.g., Gaussian noise, random gain reduction, vocalisation-band masking...). The goal is for model to learn appropriate uncertainty as SNR degrades, rather than failing catastrophically at the first sign of bad signal. This is also what defines the degradation spectrum in practice, I can't evaluate on it if I haven't trained on something like it. The modality bias problem this can cause is documented in Dai et al. (2024), worth keeping in mind. 
### 4. Cascade 
train a microphone only Vib2Soun first, then build full model on top. When the acceler. is absent, the model routes through the audio-only path, which is frozen. The robustness guarantee is architectural, the model literally cannot do worse than audio-only because that path doesn't change. Inspired by Cheng et al. (2023), who showed this is the only consistently robust strategy across architectures. Main value is experiemntal since it gives a clean answer to "how much is the acceler. actually contirbuting" at every point on the degradation spectrum. I'm not expecting the audio-only floor to be high, since the ablation already suggests it won't be, but that's an honest result 

