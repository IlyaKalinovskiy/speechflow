form Fade
	real Fade_in_(s) 0.05
	real Fade_out_(s) 0 (= no change)
	boolean Preview_(click_Apply._Uncheck_to_publish) 1
endform

include batch.praat

procedure action
	s = selected("Sound")
	s$ = selected$("Sound")
	dur = Get total duration
	stt = Get start time
	et = Get end time
	fade_in = min(max(fade_in, 0), dur)
	fade_out = min(max(fade_out, 0), dur)

	Copy: s$ + "-fade"

	if fade_in > 0
		Fade in: 0, stt, fade_in, "yes"
	endif
	if fade_out > 0
		Fade out: 0, et, -fade_out, "yes"
	endif

	if preview
		result = selected("Sound")
		if stt <> 0
			Scale times to: 0, dur
		endif
		preview_dur = min(3, dur)
		pre = Extract part: 0, preview_dur, "rectangular", 1, "no"
		nowarn Fade in: 0, 0, 0.025, "yes"
		nowarn Fade out: 0, preview_dur, -0.025, "yes"
		Play
		selectObject: s
		removeObject: pre, result
	endif
endproc
