form Copy intensity (average dB)
	boolean Avoid_clipping 1
	boolean Preview_(click_Apply._Uncheck_to_publish) 1
endform

if numberOfSelected("Sound") = 2
	s1 = selected("Sound")
	s1$ = selected$("Sound")
	s2 = selected("Sound", 2)
	s2$ = selected$("Sound", 2)

	selectObject: s1
	int = Get intensity (dB)

	selectObject: s2
	result = Copy: "tmp"
	Scale intensity: int

	if avoid_clipping
		runScript: "declip.praat"
	endif

	if preview
include preview.inc
		selectObject: s1, s2
		removeObject: trimmed, pre, result
	else
		Rename: s2$ + "-copyintensityaverage-" + s1$
	endif
endif
