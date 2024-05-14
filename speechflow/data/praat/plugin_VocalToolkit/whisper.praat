include batch.praat

procedure action
	s$ = selected$("Sound")
	original_dur = Get total duration
	int = Get intensity (dB)

	if int <> undefined
		runScript: "workpre.praat"
		tmp1 = selected("Sound")
		sf = Get sampling frequency
		dur = Get total duration
		Scale peak: 0.99

		runScript: "gate.praat", -40, 0.1, 0.05, 0, 0, 0
		tmp2 = selected("Sound")
		Formula: "self + randomUniform(-0.00001, 0.00001)"

		pred_order = round(sf / 1000) + 2
		lpc = noprogress To LPC (burg): pred_order, 0.025, 0.01, 50

		noise = Create Sound from formula: "noise", 1, 0, dur, sf, "randomUniform(-1, 1)"
		plusObject: lpc
		tmp3 = Filter: "yes"

		runScript: "workpost.praat", original_dur
		tmp4 = selected("Sound")

		runScript: "eq10bands.praat", -24, -24, -24, -24, 12, 24, 24, 12, 12, -6, 0
		tmp5 = selected("Sound")

		runScript: "gate.praat", -90, 0.1, 0.05, 0, 0, 0
		tmp6 = selected("Sound")

		Scale peak: 0.99

		intensity = noprogress To Intensity: 400, 0, "no"
		Formula: "if round(self) = 0 then 0 else if self > 85 then 1 / self - (self - 85) else 1 / self fi fi"
		intensitytier = Down to IntensityTier
		plusObject: tmp6
		Multiply: "yes"

		runScript: "fixdc.praat"
		Scale intensity: int
		runScript: "declip.praat"

		removeObject: tmp1, tmp2, lpc, noise, tmp3, tmp4, tmp5, tmp6, intensity, intensitytier
	else
		Copy: "tmp"
	endif

	Rename: s$ + "-whisper"
endproc
