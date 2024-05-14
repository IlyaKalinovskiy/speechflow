if preview
	trimmed_tmp = nocheck nowarn Trim silences: 0.08, "yes", 100, 0, -35, 0.1, 0.05, "no", "trimmed"
	if trimmed_tmp = s or trimmed_tmp = undefined
		trimmed = Copy: "tmp"
	else
		trimmed = trimmed_tmp
	endif
	trimmed_dur = Get total duration
	stt = Get start time
	if stt <> 0
		Scale times to: 0, trimmed_dur
	endif
	preview_dur = min(3, trimmed_dur)
	original_dur = preview_dur
	pre1 = Extract part: 0, preview_dur, "rectangular", 1, "no"
endif
