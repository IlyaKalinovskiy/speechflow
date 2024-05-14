form Cut pauses
	boolean Only_at_start_and_end 0
	boolean Preview_(click_Apply._Uncheck_to_publish) 1
endform

include batch.praat

procedure action
	s = selected("Sound")
	s$ = selected$("Sound")

	ch = Get number of channels
	if ch = 1
		wrk = Copy: "wrk"
	else
		wrk = Extract one channel: 1
	endif
	runScript: "fixdc.praat"

	trimmed = nocheck nowarn Trim silences: 0.08, only_at_start_and_end, 100, 0, -35, 0.1, 0.05, "no", "trimmed"
	if trimmed = wrk or trimmed = undefined
		Copy: "tmp"
	else
		stt = Get start time
		if stt <> 0
			dur = Get total duration
			Scale times to: 0, dur
		endif
	endif

	if ch <> 1
		tmp = selected("Sound")
		Convert to stereo
		removeObject: tmp
	endif
	removeObject: wrk

	if preview
		result = selected("Sound")
		final_dur = Get total duration
		preview_dur = min(3, final_dur)
		pre = Extract part: 0, preview_dur, "rectangular", 1, "no"
		nowarn Fade in: 0, 0, 0.025, "yes"
		nowarn Fade out: 0, preview_dur, -0.025, "yes"
		Play
		selectObject: s
		removeObject: pre, result
	else
		Rename: s$ + "-cutpauses"
	endif
endproc
