# Vocal tract size change was adapted from the script "VTchange" by Chris Darwin, https://groups.io/g/Praat-Users-List/files/Darwin%20scripts

form Change vocal tract size, pitch and duration
	positive Formant_shift_ratio 1.0 (= no change)
	real New_pitch_median_(Hz) 0.0 (= no change)
	real Pitch_variation_(%) 100 (= no change)
	real New_duration_(s) 0.0 (= no change)
	boolean Preview_(click_Apply._Uncheck_to_publish) 1
endform

formant_shift_ratio = min(formant_shift_ratio, 3)
new_pitch_median = max(new_pitch_median, 0)
pitch_variation_factor = pitch_variation / 100
new_duration = max(new_duration, 0)

include batch.praat

procedure action
	s = selected("Sound")
	s$ = selected$("Sound")
	original_dur = Get total duration

	if formant_shift_ratio <> 1 or new_pitch_median <> 0 or pitch_variation <> 100 or (new_duration <> 0 and new_duration <> object[s].xmax - object[s].xmin)
		runScript: "workpre.praat"
		wrk = selected("Sound")
		dur = Get total duration
		sf = Get sampling frequency

		if new_duration = 0
			new_dur = dur
		else
			new_dur = new_duration + 0.025 + 0.025
			original_dur = new_duration
		endif
		duration_factor = new_dur / dur

		if formant_shift_ratio > 1
			rdur = formant_shift_ratio
		elsif formant_shift_ratio < 1
			rdur = 1 / (1 - formant_shift_ratio + 1)
		elsif formant_shift_ratio = 1
			rdur = 1
		endif

		if rdur * duration_factor > 3
			wrk2 = Extract part: 0, new_dur, "rectangular", 1, "no"
			dur = new_dur
		else
			wrk2 = Copy: "wrk2"
		endif

include minmaxf0.praat

		pitch = noprogress To Pitch: 0.01, minF0, maxF0
		f0 = Get quantile: 0, 0, 0.50, "Hertz"

		if f0 <> undefined
			plusObject: wrk2
			manipulation = noprogress To Manipulation

			if formant_shift_ratio > 1
				formula$ = "self / formant_shift_ratio"
				rsf = sf / formant_shift_ratio
			elsif formant_shift_ratio < 1
				formula$ = "self * (1 - formant_shift_ratio + 1)"
				rsf = sf * (1 - formant_shift_ratio + 1)
			endif

			@manipulate
			result = selected("Sound")
		else
			selectObject: s
			result = Copy: "tmp"
		endif

		removeObject: wrk, wrk2, pitch
	else
		result = Copy: "tmp"
	endif

	if preview
include preview.inc
		selectObject: s
		removeObject: trimmed, pre, result
	else
		Rename: s$ + "-changevtsizepitchduration"
	endif
endproc

procedure manipulate
	.pitchtier = Extract pitch tier

	if formant_shift_ratio <> 1
		Formula: formula$
	endif

	if new_pitch_median <> 0
		.f0_f = new_pitch_median / f0
		Formula: "self * .f0_f"
	endif

	if pitch_variation_factor <> 1
		if new_pitch_median = 0
			.pm = number(fixed$(f0, 3))
		else
			.pm = number(fixed$(new_pitch_median, 3))
		endif

		if formant_shift_ratio > 1
			.pm = .pm / formant_shift_ratio
		elsif formant_shift_ratio < 1
			.pm = .pm * (1 - formant_shift_ratio + 1)
		endif

		.fref_st = 12 * ln(.pm / 100) / ln(2)
		Formula: "if self <> undefined then 100 * exp((.fref_st + 12 * ln(self / .pm) / ln(2) * pitch_variation_factor) * ln(2) / 12) else self fi"
	endif

	plusObject: manipulation
	Replace pitch tier

	.durationtier = Create DurationTier: "tmp", 0, dur
	Add point: 0, rdur * duration_factor
	plusObject: manipulation
	Replace duration tier

	selectObject: manipulation
	.res = Get resynthesis (overlap-add)

	if formant_shift_ratio <> 1
		.rs = Resample: rsf, 10
		Override sampling frequency: sf
	endif

	.dur2 = Get total duration
	if .dur2 <> new_dur
		.tmp = Extract part: 0, new_dur, "rectangular", 1, "no"
	endif

	runScript: "workpost.praat", original_dur

	removeObject: manipulation, .pitchtier, .durationtier, .res
	if formant_shift_ratio <> 1
		removeObject: .rs
	endif
	if .dur2 <> new_dur
		removeObject: .tmp
	endif
endproc
