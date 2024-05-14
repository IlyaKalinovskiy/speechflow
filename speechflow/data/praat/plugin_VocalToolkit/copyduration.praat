form Copy duration
	choice Method 1
		button Stretch
		button Cut or add time
	boolean Preview_(click_Apply._Uncheck_to_publish) 1
endform

if numberOfSelected("Sound") = 2
	s1 = selected("Sound")
	s1$ = selected$("Sound")
	s2 = selected("Sound", 2)
	s2$ = selected$("Sound", 2)

	selectObject: s1
	dur1 = Get total duration

	selectObject: s2
	dur2 = Get total duration

	wrk = Copy: "wrk"

	if dur1 <> dur2
		if method = 1
			runScript: "changeduration.praat", dur1, "Stretch", 0
		elsif method = 2
			runScript: "fixdc.praat"
			Extract part: 0, dur1, "rectangular", 1, "no"
		endif
		removeObject: wrk
	endif
	result = selected("Sound")

	if preview
include preview.inc
		selectObject: s1, s2
		removeObject: trimmed, pre, result
	else
		Rename: s2$ + "-copyduration-" + s1$
	endif
endif
