form Limiter
	real Threshold_(dB) 80
	boolean Preview_(click_Apply._Uncheck_to_publish) 1
endform

threshold = min(max(threshold, 0), 100)

include batch.praat

procedure action
	s = selected("Sound")
	s$ = selected$("Sound")
	original_dur = Get total duration
	int = Get intensity (dB)

	if int <> undefined

include preview1.inc

		runScript: "workpre.praat"
		wrk = selected("Sound")

		intensity = noprogress To Intensity: 400, 0, "no"
		Formula: "if round(self) = 0 then 0 else if self > threshold then 1 / self - (self - threshold) else 1 / self fi fi"

		intensitytier = Down to IntensityTier
		plusObject: wrk
		tmp = Multiply: "yes"

		runScript: "workpost.praat", original_dur
		Scale intensity: int
		runScript: "declip.praat"
		result = selected("Sound")

		removeObject: wrk, intensity, intensitytier, tmp

include preview2.inc

		if not preview
			Rename: s$ + "-limiter_" + string$(threshold)
		endif
	else
		if not preview
			Copy: s$ + "-limiter_" + string$(threshold)
		endif
	endif
endproc
