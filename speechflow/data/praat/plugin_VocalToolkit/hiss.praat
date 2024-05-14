form Hiss filter (low-pass)
	real Frequency_(1000-20000_Hz) 7500
	boolean Preview_(click_Apply._Uncheck_to_publish) 1
endform

frequency = min(max(frequency, 1000), 20000)

include batch.praat

procedure action
	s = selected("Sound")
	s$ = selected$("Sound")
	int = Get intensity (dB)

	if int <> undefined

include preview1.inc

		wrk = Copy: "wrk"
		runScript: "fixdc.praat"
		Filter (pass Hann band): 0, frequency, 100
		runScript: "fixdc.praat"
		result = selected("Sound")
		removeObject: wrk

include preview2.inc

		if not preview
			Rename: s$ + "-rumblefilter_" + string$(frequency)
		endif
	else
		if not preview
			Copy: s$ + "-rumblefilter_" + string$(frequency)
		endif
	endif
endproc
