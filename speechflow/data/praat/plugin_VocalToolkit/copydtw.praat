form Dynamic time warping (DTW)
	choice Slope_constraint 3
		button no restriction
		button 1/3 < slope < 3
		button 1/2 < slope < 2
		button 2/3 < slope < 3/2
	boolean Preview_(click_Apply._Uncheck_to_publish) 1
endform

if numberOfSelected("Sound") = 2
	s1 = selected("Sound")
	s1$ = selected$("Sound")
	s2 = selected("Sound", 2)
	s2$ = selected$("Sound", 2)

	selectObject: s1
	original_dur = Get total duration
	int1 = Get intensity (dB)

	selectObject: s2
	int2 = Get intensity (dB)

	if int1 <> undefined and int2 <> undefined
		sf2 = Get sampling frequency

		runScript: "workpre.praat"
		wrk1 = selected("Sound")
		dur1 = Get total duration

		selectObject: s1
		sf1 = Get sampling frequency

		runScript: "workpre.praat"
		if sf1 <> sf2
			tmp1 = selected("Sound")
			Resample: sf2, 50
			removeObject: tmp1
		endif
		wrk2 = selected("Sound")
		dur2 = Get total duration

		plusObject: wrk1
		time_step = 0.005
		dtw = noprogress To DTW: 0.015, time_step, 0.1, slope_constraint$

		n_frames = ceiling(dur2 / time_step)
		x_time = 0

		for i to n_frames + 1
			y_time[i] = Get y time from x time: x_time
			x_time += time_step
		endfor

		for i to n_frames
			time_r[i] = time_step / (y_time[i + 1] - y_time[i])
		endfor

		durationtier = Create DurationTier: "tmp", 0, dur1
		for i to n_frames
			if i = 1
				Add point: y_time[i], time_r[i]
			else
				if number(fixed$(time_r[i], 6)) <> number(fixed$(time_r[i - 1], 6))
					Add point: y_time[i] + 0.00000000001, time_r[i]
				endif
			endif
			if i = n_frames
				Add point: y_time[i + 1], time_r[i]
			else
				if number(fixed$(time_r[i], 6)) <> number(fixed$(time_r[i + 1], 6))
					Add point: y_time[i + 1], time_r[i]
				endif
			endif
		endfor

		selectObject: wrk1

include minmaxf0.praat

		pitch = noprogress To Pitch: 0.01, minF0, maxF0
		plusObject: wrk1
		manipulation = noprogress To Manipulation

		plusObject: durationtier
		Replace duration tier

		selectObject: manipulation
		res = Get resynthesis (overlap-add)

		runScript: "workpost.praat", original_dur
		runScript: "declip.praat"
		result = selected("Sound")

		removeObject: wrk1, wrk2, dtw, durationtier, pitch, manipulation, res
	else
		result = Copy: s2$ + "-dtw-" + s1$
	endif

	if preview
include preview.inc
		selectObject: s1, s2
		removeObject: trimmed, pre, result
	else
		Rename: s2$ + "-dtw-" + s1$
	endif
endif
