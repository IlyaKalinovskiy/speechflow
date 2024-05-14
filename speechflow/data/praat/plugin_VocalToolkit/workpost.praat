form workpost
	real original_dur
endform

stt = Get start time
et = Get end time

Extract part: stt + 0.025, et - 0.025, "rectangular", 1, "no"
runScript: "fixdc.praat"
final_dur = Get total duration

if final_dur <> original_dur
	tmp = selected("Sound")
	Extract part: 0, original_dur, "rectangular", 1, "no"
	removeObject: tmp
endif
