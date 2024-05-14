form Change duration
	positive New_duration_(s) 3.0
	choice Method 1
		button Stretch
		button Cut or add time
	boolean Preview_(click_Apply._Uncheck_to_publish) 1
endform

include batch.praat

procedure action
	s = selected("Sound")
	s$ = selected$("Sound")
	dur = Get total duration

	if dur <> new_duration
		if method = 1
			wrk = Copy: "wrk"
			runScript: "fixdc.praat"

include minmaxf0.praat

			duration_factor = new_duration / dur

			if duration_factor > 3
				wrk2 = Extract part: 0, new_duration, "rectangular", 1, "no"
				dur = new_duration
			else
				wrk2 = Copy: "wrk2"
			endif

			pitch = noprogress To Pitch: 0.01, minF0, maxF0
			f0 = Get quantile: 0, 0, 0.50, "Hertz"

			if f0 <> undefined
				plusObject: wrk2
				manipulation = noprogress To Manipulation

				durationtier = Create DurationTier: "tmp", 0, dur
				Add point: 0, duration_factor
				plusObject: manipulation
				Replace duration tier

				selectObject: manipulation
				res = Get resynthesis (overlap-add)

				dur2 = Get total duration
				if dur2 <> new_duration
					tmp = selected("Sound")
					Extract part: 0, new_duration, "rectangular", 1, "no"
					removeObject: tmp
				endif

				runScript: "fixdc.praat"
				result = selected("Sound")
				removeObject: wrk, wrk2, pitch, durationtier, manipulation
			else
				selectObject: s
				result = Copy: "tmp"
				removeObject: wrk, wrk2, pitch
			endif

		elsif method = 2
			stt = Get start time
			Extract part: stt, stt + new_duration, "rectangular", 1, "no"
			result = selected("Sound")
		endif
	else
		result = Copy: "tmp"
	endif

	if preview
include preview.inc
		selectObject: s
		removeObject: trimmed, pre, result
	else
		Rename: s$ + "-changeduration_" + method$ + "__" + string$(new_duration)
	endif
endproc
