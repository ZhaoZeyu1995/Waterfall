class MaskWaveform(object):
    def __init__(
        self,
        mask_factor: int = 5,
    ):
        self.mask_factor = mask_factor

    def __call__(self, sample):
        assert "wav" in sample.keys(), "wav is not in sample"
        assert "uttid" in sample.keys(), "uttid is not in sample"
        wav = sample["wav"]
        wav_len = sample["wav_len"]
        uttid = sample["uttid"]
        mask_part = int(uttid.split("_")[-1])
        assert (
            mask_part < self.mask_factor
        ), "mask_part should be less than mask_factor, but got {} and {}".format(
            mask_part, self.mask_factor
        )
        # mask the mask_part / mask_factor part of the waveform
        mask_start = int(wav_len * mask_part / self.mask_factor)
        mask_end = int(wav_len * (mask_part + 1) / self.mask_factor)
        wav[mask_start:mask_end] = 0
        wav = (wav - wav.mean()) / torch.sqrt(wav.var())  # Normalisation
        sample["wav"] = wav
        return sample
