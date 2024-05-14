form Add noise
	real Volume_(dB) 40
	choice Type 1
		button White noise
		button Pink noise
		button Brown noise
	boolean Preview_(click_Apply._Uncheck_to_publish) 1
endform

volume = min(max(round(volume), 1), 100)

include batch.praat

procedure action
	s = selected("Sound")
	s$ = selected$("Sound")
	sf = Get sampling frequency
	ch = Get number of channels
	ch = min(max(ch, 1), 2)
	st = ch - 1

include preview1.inc

	result = Copy: "tmp"
	runScript: "fixdc.praat"
	dur = Get total duration

	runScript: "createwaveform.praat", dur, sf, 140, 0.2, 0, st, type$, 0
	noise = selected("Sound")
	Scale intensity: volume

	selectObject: result
	Formula: "self + object[noise]"
	runScript: "declip.praat"

	removeObject: noise

include preview2.inc

	if not preview
		Rename: s$ + "-add_" + type$ + "__" + string$(volume)
	endif
endproc
