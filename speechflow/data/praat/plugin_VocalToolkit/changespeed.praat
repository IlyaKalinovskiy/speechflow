form Change speed
	comment Speeds up or slows down the selected Sounds, affecting duration and pitch.
	choice Change_by 1
		button Factor
		button New duration
		button Frame rate conversion (video fps)
	positive Factor 0.5
	positive New_duration_(s) 2
	comment "Frame rate conversion" can be used to convert the audio track of a video
	positive Original_frame_rate_(fps) 25 (= PAL)
	positive New_frame_rate_(fps) 23.976 (= NTSC)
	boolean Preview_(click_Apply._Uncheck_to_publish) 1
endform

include batch.praat

procedure action
	s = selected("Sound")
	s$ = selected$("Sound")
	sf = Get sampling frequency
	dur = Get total duration

	if change_by = 1
		new_sf = sf * factor
		name$ = s$ + "-change_speed__factor_" + string$(factor)
	elsif change_by = 2
		new_sf = sf * (dur / new_duration)
		name$ = s$ + "-change_speed__new_duration_" + string$(new_duration)
	elsif change_by = 3
		new_sf = sf * (new_frame_rate / original_frame_rate)
		name$ = s$ + "-change_speed__fps_" + string$(new_frame_rate)
	endif

	wrk = Copy: "wrk"
	Override sampling frequency: new_sf
	result = Resample: sf, 50

	removeObject: wrk

	if preview
include preview.inc
		selectObject: s
		removeObject: trimmed, pre, result
	else
		Rename: name$
	endif
endproc
