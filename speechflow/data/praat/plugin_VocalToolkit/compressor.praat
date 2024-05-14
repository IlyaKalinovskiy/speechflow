form Compressor
	real Compression_(%) 25
	boolean Preview_(click_Apply._Uncheck_to_publish) 1
endform

compression = min(max(compression, 0), 100)

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
		Formula: "-self * compression / 100"

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
			Rename: s$ + "-compressor_" + string$(compression)
		endif
	else
		if not preview
			Copy: s$ + "-compressor_" + string$(compression)
		endif
	endif
endproc
