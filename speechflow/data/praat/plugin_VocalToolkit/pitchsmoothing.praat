form Pitch smoothing
	real Pitch_smoothing_(%) 50
	boolean Preview_(click_Apply._Uncheck_to_publish) 1
endform

pitch_smoothing = min(max(pitch_smoothing, 0), 100)

include batch.praat

procedure action
	s = selected("Sound")
	s$ = selected$("Sound")
	dur = Get total duration

include preview1.inc

	if pitch_smoothing <> 0
		runScript: "workpre.praat"
		wrk = selected("Sound")

include minmaxf0.praat

		pitch = noprogress To Pitch: 0.01, minF0, maxF0
		f0 = Get quantile: 0, 0, 0.50, "Hertz"

		if f0 <> undefined
			plusObject: wrk
			manipulation = noprogress To Manipulation

			selectObject: pitch
			smoothedpitch = Smooth: -6 * ln(pitch_smoothing / 10) + 15

			pitchtier = Down to PitchTier
			plusObject: manipulation
			Replace pitch tier

			selectObject: manipulation
			res = Get resynthesis (overlap-add)

			runScript: "workpost.praat", dur
			result = selected("Sound")
			removeObject: wrk, pitch, manipulation, smoothedpitch, pitchtier, res
		else
			selectObject: s
			result = Copy: "tmp"
			removeObject: wrk, pitch
		endif
	else
		result = Copy: "tmp"
	endif

include preview2.inc

	if not preview
		Rename: s$ + "-pitchsmoothing_" + string$(pitch_smoothing)
	endif
endproc
