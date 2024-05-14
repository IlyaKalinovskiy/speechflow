form Shift frequencies
	comment Shifts all frequencies by the same amount.
	real Shift_by_(Hz) 1000
	real New_sampling_frequency_(Hz) 44100
	boolean Preview_(click_Apply._Uncheck_to_publish) 1
endform

include batch.praat

procedure action
	s = selected("Sound")
	s$ = selected$("Sound")
	dur = Get total duration
	sf = Get sampling frequency
	int = Get intensity (dB)

	if int <> undefined

include preview1.inc

		if new_sampling_frequency = 0
			nsf = sf
		else
			nsf = new_sampling_frequency
		endif
		sp = To Spectrum: "yes"
		sp_sh = Shift frequencies: shift_by, nsf / 2, 50
		tmp = To Sound
		result = Extract part: 0, dur, "rectangular", 1, "no"
		runScript: "declip.praat"

		removeObject: sp, sp_sh, tmp

include preview2.inc

		if not preview
			Rename: s$ + "-frequencyshifted"
		endif
	else
		if not preview
			Copy: s$ + "-frequencyshifted"
		endif
	endif
endproc
